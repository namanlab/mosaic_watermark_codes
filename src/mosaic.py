"""
MOSAIC: a set-based language-model watermark.

Where every prior watermark (KGW, the spectral/FFT scheme, PULSAR) reads the
token stream as an ordered *sequence*, MOSAIC treats it as an unordered *bag of
parity samples*. The pipeline:

  embed:
    for each generated position that is "active" (model entropy above a gate):
      - from the order-canonical local context (sorted predecessor window),
        derive a fountain check: a content-chosen subset D of message-bit
        indices, plus a chip bit r, plus a green mask G.
      - target parity g = (XOR_{i in D} m_i) XOR r
      - add +delta to the logits of G if g==1 else of its complement, sample.

  detect (model-free):
    for each position, recompute D, r, G from its context; observe the token's
    color; descramble with r -> a soft observation (LLR) of the parity over D.
    This is one fountain check. Collect ALL checks (order doesn't matter) and
    run belief propagation to recover m. Report a union-bound p-value.

Why this is robust:
  * rearrangement: a check depends only on a position's predecessor multiset,
    so reordering blocks of text leaves interior checks unchanged.
  * deletion / length change: the code is rateless -- fewer tokens just means
    fewer checks; BP decodes from whatever survives.
  * substitution: each token contributes to a parity over several message bits,
    spreading any single corruption.

The decoder is belief propagation (order-agnostic), NOT a trellis -- that choice
is what lets the watermark be a set rather than a sequence.
"""

import math
from typing import Dict, List, Optional

import logging

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .content_prf import ContentPRF
from .fountain_bp import robust_soliton, bp_decode

# Library logger. A NullHandler keeps direct use silent unless the application
# configures logging (e.g. run_experiments.py adds console + file handlers).
log = logging.getLogger("mosaic")
log.addHandler(logging.NullHandler())


class MosaicWatermark:
    def __init__(
        self,
        model_name: str = "gpt2",
        secret_key: str = "mosaic_key",
        k_bits: int = 32,
        delta: float = 6.0,
        gamma: float = 0.5,
        temperature: float = 0.8,
        top_p: float = 0.95,
        window: int = 2,
        max_degree: int = 8,
        soft_mag: float = 4.0,
        device: Optional[str] = None,
        load_model: bool = True,
    ):
        self.secret_key = secret_key
        self.k_bits = k_bits
        self.delta = delta
        self.gamma = gamma
        self.temperature = temperature
        self.top_p = top_p          # nucleus cutoff; <=0 or >=1 disables filtering
        self.window = window
        self.max_degree = max_degree
        self.soft_mag = soft_mag  # detection LLR magnitude A = ln((1-pe)/pe)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else (
                "mps" if torch.backends.mps.is_available() else "cpu")
        self.device = device

        log.info("loading tokenizer: %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.vocab_size = len(self.tokenizer)

        self.model = None
        if load_model:
            log.info("loading model: %s on %s", model_name, device)
            kwargs = {}
            if device in ("cuda", "mps"):
                kwargs["torch_dtype"] = torch.float16
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
            self.model.to(device)
            self.model.eval()
            log.info("model loaded (%.0fM params)",
                     sum(p.numel() for p in self.model.parameters()) / 1e6)

        self.degree_dist = robust_soliton(k_bits, c=0.03, delta=0.5)
        self.prf = ContentPRF(secret_key, self.vocab_size, k_bits, gamma,
                              window, self.degree_dist, max_degree)

    def reconfigure(self, k_bits=None, delta=None, window=None, soft_mag=None):
        """
        Change payload size / bias / window / detection magnitude in place,
        WITHOUT reloading the model. Lets one model instance serve many
        experimental settings (essential for large models, where holding
        several instances would exhaust memory).
        """
        if delta is not None:
            self.delta = delta
        if soft_mag is not None:
            self.soft_mag = soft_mag
        rebuild = False
        if k_bits is not None and k_bits != self.k_bits:
            self.k_bits = k_bits
            self.degree_dist = robust_soliton(k_bits, c=0.03, delta=0.5)
            rebuild = True
        if window is not None and window != self.window:
            self.window = window
            rebuild = True
        if rebuild:
            self.prf = ContentPRF(self.secret_key, self.vocab_size, self.k_bits,
                                  self.gamma, self.window, self.degree_dist,
                                  self.max_degree)

    # ------------------------------------------------------------------
    def _sample(self, logits) -> int:
        """Pick the next token from the (already watermark-biased) logits.

        Matches the segment-watermark / Qu et al (arXiv:2401.16820) decoding:
        temperature scaling, then nucleus (top-p) truncation, then a multinomial
        draw. temperature <= 0 falls back to greedy argmax. The top-p cutoff
        discards the long low-probability tail (including the many inflated
        green-list tokens) BEFORE sampling, which is what keeps perplexity low at
        large delta; plain temperature-1.0 multinomial sampling instead reaches
        deep into that tail and produces degenerate text.
        """
        if self.temperature is None or self.temperature <= 0.0:
            return int(torch.argmax(logits).item())
        probs = torch.softmax(logits / self.temperature, dim=0)
        if self.top_p is not None and 0.0 < self.top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumprobs = torch.cumsum(sorted_probs, dim=0)
            # keep the smallest prefix whose cumulative mass reaches top_p
            # (the top token is always kept); zero the rest and renormalise
            mask = cumprobs - sorted_probs > self.top_p
            sorted_probs[mask] = 0.0
            sorted_probs.div_(sorted_probs.sum())
            choice = int(torch.multinomial(sorted_probs, 1).item())
            return int(sorted_idx[choice].item())
        return int(torch.multinomial(probs, 1).item())

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, prompt: str, message: List[int],
                 max_new_tokens: int = 200, watermark: bool = True) -> Dict:
        """
        Generate text from `prompt`.

        watermark=True  embeds `message` (the bit marking).
        watermark=False produces a plain, unmarked baseline from the same model
                        and settings, for quality comparison. `message` is then
                        ignored (pass any k-bit list, or the same one).
        """
        assert self.model is not None
        if watermark:
            assert len(message) == self.k_bits
            m = np.array(message, dtype=np.int8)

        enc = self.tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)
        prompt_len = input_ids.shape[1]

        generated = input_ids[0].tolist()
        past = None
        cur = input_ids
        eos = self.tokenizer.eos_token_id

        for _ in range(max_new_tokens):
            out = self.model(cur, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[0, -1, :].float().cpu()[: self.vocab_size]
            if eos is not None and eos < self.vocab_size:
                logits[eos] = -1e10

            if watermark:
                # Embed a fountain check at every position (constant +delta on a
                # content-keyed half-vocabulary -> identical quality profile to a
                # plain KGW watermark). The detector reproduces every check, so
                # there is no active/inactive ambiguity.
                subset = self.prf.check_subset(generated)
                r = self.prf.chip(generated)
                parity = int(np.bitwise_xor.reduce(m[subset]))
                g = parity ^ r
                mask = self.prf.green_mask(generated)
                bias = torch.from_numpy(
                    np.where(mask == bool(g), self.delta, 0.0)).float()
                logits = logits + bias

            nxt = self._sample(logits)
            generated.append(nxt)
            cur = torch.tensor([[nxt]], device=self.device)

        gen_ids = generated[prompt_len:]
        return {
            "text": self.tokenizer.decode(generated, skip_special_tokens=False),
            "generated_text": self.tokenizer.decode(gen_ids, skip_special_tokens=False),
            "generated_ids": gen_ids,
            "prompt_ids": generated[:prompt_len],
            "prompt_length": prompt_len,
            "message": list(message) if watermark else None,
            "watermarked": watermark,
        }

    @torch.no_grad()
    def perplexity(self, prompt: str, generated_ids: List[int]) -> float:
        """
        Perplexity of the generated continuation under the *base* (unbiased)
        model, given the prompt as context. A standard fluency proxy: lower is
        more natural. Computed with one forward pass, no watermark bias, so it
        scores watermarked and plain text on the same footing.
        """
        assert self.model is not None
        prompt_ids = self.tokenizer(prompt)["input_ids"]
        full = list(prompt_ids) + list(generated_ids)
        if len(generated_ids) == 0:
            return float("nan")
        ids = torch.tensor([full], device=self.device)
        logits = self.model(ids).logits[0].float()           # [L, V]
        logprobs = torch.log_softmax(logits[:-1], dim=-1)    # predicts full[1:]
        targets = torch.tensor(full[1:], device=self.device)
        tok_lp = logprobs[torch.arange(len(targets)), targets]
        start = len(prompt_ids)                              # first generated pos
        gen_lp = tok_lp[start - 1:]                           # scores generated toks
        nll = -gen_lp.mean().item()
        return float(math.exp(nll))

    # ------------------------------------------------------------------
    def _collect_checks(self, token_ids, prompt_ids):
        """
        Re-derive every position's fountain check from the text. Returns the
        list of (subset, llr) checks and the per-position raw data needed for
        scoring.
        """
        # Detection LLR magnitude A (= ln((1-pe)/pe) for the assumed average
        # error pe). Configurable via self.soft_mag so experiments can sweep it.
        soft_mag = self.soft_mag
        checks = []
        records = []
        context = list(prompt_ids) if prompt_ids else \
            [self.tokenizer.bos_token_id or 0]
        for tok in token_ids:
            tok = int(tok)
            # Every position carries a check (embedder biased every position).
            subset = self.prf.check_subset(context)
            r = self.prf.chip(context)
            mask = self.prf.green_mask(context)
            in_green = bool(mask[tok]) if tok < self.vocab_size else False
            # observed parity = (color) XOR-descrambled by chip
            obs_parity = (1 if in_green else 0) ^ r
            llr = (2 * obs_parity - 1) * soft_mag
            checks.append((subset.tolist(), llr))
            records.append((tuple(int(s) for s in subset), r, in_green))
            context.append(tok)
        return checks, records

    def detect_ids(self, token_ids, prompt_ids=None,
                   max_iters: int = 80) -> Dict:
        checks, records = self._collect_checks(token_ids, prompt_ids)
        est = bp_decode(self.k_bits, checks, max_iters=max_iters)
        est_list = [int(b) for b in est]

        # Score: how many positions' observed color agree with the decoded
        # message's predicted target. Under H0 this is Binomial(T, 1/2).
        S, T = 0, len(records)
        for subset, r, in_green in records:
            parity = int(np.bitwise_xor.reduce(est[list(subset)])) if subset else 0
            g = parity ^ r
            if in_green == bool(g):
                S += 1
        p_value = self._union_bound_pvalue(T, S, self.k_bits)
        return {
            "message": est_list,
            "score": S,
            "num_tokens": T,
            "match_fraction": S / max(T, 1),
            "p_value": p_value,
            "detected": p_value < 0.01,
        }

    def detect_text(self, text: str, prompt: Optional[str] = None,
                    max_iters: int = 80) -> Dict:
        if prompt is not None:
            prompt_ids = self.tokenizer(prompt)["input_ids"]
            full = self.tokenizer(text)["input_ids"]
            gen_ids = full[len(prompt_ids):]
            return self.detect_ids(gen_ids, prompt_ids=prompt_ids, max_iters=max_iters)
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        return self.detect_ids(ids, prompt_ids=None, max_iters=max_iters)

    def _union_bound_pvalue(self, T: int, S: int, k: int) -> float:
        if T == 0:
            return 1.0
        p0 = 0.5
        if T <= 1000:
            log_terms = [
                math.lgamma(T + 1) - math.lgamma(s + 1) - math.lgamma(T - s + 1)
                + s * math.log(p0) + (T - s) * math.log(1 - p0)
                for s in range(S, T + 1)
            ]
            mx = max(log_terms)
            log_tail = mx + math.log(sum(math.exp(lt - mx) for lt in log_terms))
        else:
            from math import erfc, sqrt
            z = (S - 0.5 - T * p0) / sqrt(T * p0 * (1 - p0))
            log_tail = math.log(max(0.5 * erfc(z / sqrt(2)), 1e-300))
        log_p = k * math.log(2) + log_tail
        return min(1.0, math.exp(min(log_p, 0.0)))
