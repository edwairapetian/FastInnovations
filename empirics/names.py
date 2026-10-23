"""Name keys for name-keyed identities (no external ids)."""
import re
import unicodedata

_strip = re.compile(r"[^a-z\- ]")


def _norm(s):
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    s = re.sub(r"\\['`^\"~=.uvHc]\{?([a-z])\}?", r"\1", s)   # TeX accents
    return _strip.sub(" ", s.replace(".", " ")).split()


def submitter_key(s, full=True):
    """'Pavel M. Nadolsky' -> 'pavel nadolsky' (full) or 'p nadolsky'."""
    if not isinstance(s, str):
        return None
    if "," in s:
        last, first = s.split(",", 1)
        s = first + " " + last
    t = _norm(s)
    if not t:
        return None
    first = t[0] if full else t[0][0]
    return f"{first} {t[-1]}"


def author_keys(authors, full=False):
    """'Nadolsky, Pavel M.;Yuan, C.-P.' -> ['p nadolsky', 'c yuan']."""
    out = []
    if not isinstance(authors, str):
        return out
    for a in authors.split(";"):
        last, _, first = a.partition(",")
        lt, ft = _norm(last), _norm(first)
        if not lt:
            continue
        if not ft:
            out.append(lt[-1])
            continue
        out.append(f"{ft[0] if full else ft[0][0]} {lt[-1]}")
    return out
