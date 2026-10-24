"""Stream the "samples" array of a result JSON one object at a time.

The main result file is 147 MB and json.load turns it into several GB of Python
objects. This reader parses one sample at a time from a sliding buffer of about
one chunk, so memory stays near the size of a single chunk regardless of the
file size. Strings inside samples may contain braces; json.JSONDecoder handles
them correctly.

    from stream_samples import iter_samples
    for s in iter_samples(path, fields=("prompt_index", "k", "generated_ids")):
        ...
"""
import json

_WS = " \t\n\r,"


def iter_samples(path, fields=None, chunk=1 << 20):
    """Yield the objects of the top-level "samples" array of the JSON at path.

    fields: optional tuple of keys; each yielded dict keeps only those keys.
    """
    dec = json.JSONDecoder()
    with open(path, "r") as f:
        buf = ""
        while '"samples"' not in buf:
            more = f.read(chunk)
            if not more:
                return
            buf += more
        pos = buf.index('"samples"')
        while "[" not in buf[pos:]:
            buf += f.read(chunk)
        pos = buf.index("[", pos) + 1
        eof = False
        while True:
            # skip whitespace and separators
            while True:
                while pos < len(buf) and buf[pos] in _WS:
                    pos += 1
                if pos < len(buf) or eof:
                    break
                more = f.read(chunk)
                eof = not more
                buf = buf[pos:] + more
                pos = 0
            if pos >= len(buf) or buf[pos] == "]":
                return
            try:
                obj, end = dec.raw_decode(buf, pos)
            except json.JSONDecodeError:
                if eof:
                    raise
                more = f.read(chunk)
                eof = not more
                buf = buf[pos:] + more
                pos = 0
                continue
            pos = end
            if pos > chunk:                      # drop consumed text
                buf = buf[pos:]
                pos = 0
            yield {k: obj[k] for k in fields} if fields else obj
