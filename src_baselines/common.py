"""Shared machinery for the baseline watermarks.

Protocol pieces here mirror src/mosaic.py exactly:
  - model loading (fp16 on cuda/mps, eval mode)
  - nucleus sampling (temperature scaling -> top-p truncation -> multinomial,
    greedy fallback at temperature <= 0)
  - the generation loop (one forward pass per token, eos masked out)

The baselines differ from MOSAIC only in WHICH vocabulary subset gets the
+delta bias at each step and in how the payload is read back. All three use
the standard KGW context convention: per-token quantities are keyed on the
single previous token id (h = 1), which is the default in the reference
implementations of KGW, MPAC, and Qu et al.
"""
import hashlib
import logging

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

log = logging.getLogger("baselines")

# ---------------------------------------------------------------- PRF -------
_U64 = np.uint64


def _key64(secret: str, domain: str) -> np.uint64:
    h = hashlib.sha256(f"{secret}|{domain}".encode()).digest()
    return _U64(int.from_bytes(h[:8], "little"))


def mix64(x):
    """SplitMix64 finalizer, vectorized over uint64 numpy arrays."""
    with np.errstate(over="ignore"):
        x = (x + _U64(0x9E3779B97F4A7C15))
        x = (x ^ (x >> _U64(30))) * _U64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> _U64(27))) * _U64(0x94D049BB133111EB)
        x = x ^ (x >> _U64(31))
    return x


class PRF:
    """Keyed integer PRF over (previous token, candidate token)."""

    def __init__(self, secret: str, domain: str, vocab_size: int):
        self.key = _key64(secret, domain)
        self.vocab = np.arange(vocab_size, dtype=np.uint64)

    def over_vocab(self, prev_id: int) -> np.ndarray:
        """PRF values for every candidate token given the context token."""
        with np.errstate(over="ignore"):
            return mix64(self.key ^ mix64(_U64(prev_id)) ^ self.vocab)

    def scalar(self, prev_id: int, extra: int = 0) -> int:
        with np.errstate(over="ignore"):
            v = mix64(self.key ^ mix64(_U64(prev_id)) ^ _U64(extra))
        return int(v)

    def salted_over_vocab(self, prev_id: int, salt: int) -> np.ndarray:
        """PRF values over the vocabulary for a keyed sub-scheme `salt`."""
        with np.errstate(over="ignore"):
            return mix64(self.key ^ mix64(_U64(prev_id)) ^
                         mix64(_U64(salt)) ^ self.vocab)

    def token_over_salts(self, prev_id: int, tok: int, salts: np.ndarray) -> np.ndarray:
        """PRF values for one token across many keyed sub-schemes `salts`."""
        with np.errstate(over="ignore"):
            return mix64(self.key ^ mix64(_U64(prev_id)) ^
                         mix64(salts.astype(_U64)) ^ _U64(tok))


# ------------------------------------------------------------- language model
class BaselineLM:
    """Model + tokenizer + the MOSAIC sampling rule (tau, top-p)."""

    def __init__(self, model_name, temperature, top_p, device=None,
                 load_model=True):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else (
                "mps" if torch.backends.mps.is_available() else "cpu")
        self.device = device
        self.temperature = temperature
        self.top_p = top_p
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

    # identical to MosaicWatermark._sample
    def sample(self, logits) -> int:
        if self.temperature is None or self.temperature <= 0.0:
            return int(torch.argmax(logits).item())
        probs = torch.softmax(logits / self.temperature, dim=0)
        if self.top_p is not None and 0.0 < self.top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumprobs = torch.cumsum(sorted_probs, dim=0)
            mask = cumprobs - sorted_probs > self.top_p
            sorted_probs[mask] = 0.0
            sorted_probs.div_(sorted_probs.sum())
            choice = int(torch.multinomial(sorted_probs, 1).item())
            return int(sorted_idx[choice].item())
        return int(torch.multinomial(probs, 1).item())

    @torch.no_grad()
    def generate(self, prompt: str, wm, message, max_new_tokens: int,
                 delta: float):
        """Generate with the baseline's +delta bias applied at every step.

        wm.favored_set(prev_id, pos, message) returns a boolean numpy array over
        the vocabulary marking the tokens whose logits receive +delta. `pos` is
        the 0-indexed generation step (positional schemes use it; content-keyed
        schemes ignore it).
        """
        assert self.model is not None
        enc = self.tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)
        prompt_len = input_ids.shape[1]
        generated = input_ids[0].tolist()
        past, cur = None, input_ids
        eos = self.tokenizer.eos_token_id
        for pos in range(max_new_tokens):
            out = self.model(cur, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[0, -1, :].float().cpu()[: self.vocab_size]
            if eos is not None and eos < self.vocab_size:
                logits[eos] = -1e10
            favored = wm.favored_set(generated[-1], pos, message)
            logits = logits + torch.from_numpy(
                np.where(favored, delta, 0.0)).float()
            nxt = self.sample(logits)
            generated.append(nxt)
            cur = torch.tensor([[nxt]], device=self.device)
        gen_ids = generated[prompt_len:]
        return {
            "generated_ids": gen_ids,
            "generated_text": self.tokenizer.decode(gen_ids, skip_special_tokens=False),
            "prompt_ids": generated[:prompt_len],
        }
