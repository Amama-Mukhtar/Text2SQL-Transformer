"""Task 4: greedy decoding, beam search, parser (text -> WikiSQL dict), readable SQL."""
import re

import common  # noqa: F401  (puts starter/ on sys.path)

import torch

from data_prep import AGG_OPS, COND_OPS, MAX_COLS, encode_source
from tokenizer import BOS_ID, EOS_ID

MAX_LEN = 64  # max generated tokens


# ----------------------------------------------------------------- decoding
@torch.no_grad()
def greedy_decode(model, src, max_len=MAX_LEN):
    """src: (B, S) -> list of B lists of token ids (without <s>, without </s>)."""
    model.eval()
    B = src.size(0)
    enc_out, src_mask = model.encode(src)
    ys = torch.full((B, 1), BOS_ID, dtype=torch.long, device=src.device)
    done = torch.zeros(B, dtype=torch.bool, device=src.device)
    for _ in range(max_len):
        logits = model.decode(ys, enc_out, src_mask)
        nxt = logits[:, -1].argmax(-1)
        nxt = torch.where(done, torch.zeros_like(nxt), nxt)  # finished rows just get <pad>
        ys = torch.cat([ys, nxt.unsqueeze(1)], dim=1)
        done = done | (nxt == EOS_ID)
        if done.all():
            break
    out = []
    for row in ys[:, 1:].tolist():
        toks = []
        for t in row:
            if t == EOS_ID or t == 0:
                break
            toks.append(t)
        out.append(toks)
    return out


@torch.no_grad()
def beam_search(model, src, beam=4, max_len=MAX_LEN):
    """src: (1, S). Returns the best hypothesis as a list of ids (no <s>, no </s>).
    Hypotheses are ranked by average log-prob per token (simple length normalisation)."""
    model.eval()
    enc_out, src_mask = model.encode(src)
    beams = [([BOS_ID], 0.0)]
    finished = []
    for _ in range(max_len):
        seqs = torch.tensor([b[0] for b in beams], device=src.device)
        k = seqs.size(0)
        logits = model.decode(seqs, enc_out.expand(k, -1, -1), src_mask.expand(k, -1, -1, -1))
        logp = torch.log_softmax(logits[:, -1], dim=-1)
        cands = []
        for i, (toks, sc) in enumerate(beams):
            top = logp[i].topk(beam)
            for lp, idx in zip(top.values.tolist(), top.indices.tolist()):
                cands.append((toks + [idx], sc + lp))
        cands.sort(key=lambda c: c[1], reverse=True)
        beams = []
        for toks, sc in cands:
            if toks[-1] == EOS_ID:
                finished.append((toks, sc))
            else:
                beams.append((toks, sc))
            if len(beams) == beam:
                break
        if not beams or len(finished) >= beam:
            break
    pool = finished if finished else beams
    best = max(pool, key=lambda c: c[1] / len(c[0]))
    toks = best[0][1:]
    return [t for t in toks if t != EOS_ID]


# ------------------------------------------------------------------ parsing
_HEAD = re.compile(r"^select\s+(?:(max|min|count|sum|avg)\s+)?<c(\d+)>\s*(.*)$", re.S)
_COND = re.compile(r"^<c(\d+)>\s*([=<>])\s*(.*)$", re.S)
_SPLIT = re.compile(r"\s+and\s+(?=<c\d+>\s*[=<>])")


def parse_query(text):
    """'select count <c3> where <c1> = kim manners' -> {'sel':3,'agg':3,'conds':[[1,0,'kim manners']]}
    Returns None if the string does not follow the grammar."""
    m = _HEAD.match(text.strip())
    if not m:
        return None
    agg_name, sel, rest = m.groups()
    agg = AGG_OPS.index(agg_name.upper()) if agg_name else 0
    conds = []
    rest = rest.strip()
    if rest:
        if not rest.startswith("where "):
            return None
        for part in _SPLIT.split(rest[len("where "):]):
            m2 = _COND.match(part.strip())
            if not m2:
                return None
            col, op, val = m2.groups()
            conds.append([int(col), COND_OPS.index(op), val.strip()])
    return {"sel": int(sel), "agg": agg, "conds": conds}


def ids_to_query(sp, ids, n_cols=None):
    """ids -> parsed query dict, or None when parsing fails.
    If n_cols is given, column tokens that point past the table also count as a parse failure."""
    q = parse_query(sp.decode(ids))
    if q is None:
        return None
    if n_cols is not None:
        cols = [q["sel"]] + [c[0] for c in q["conds"]]
        if any(c >= n_cols for c in cols):
            return None
    return q


# --------------------------------------------------------------- readable SQL
def to_readable_sql(q, header):
    """parsed query + real column names -> SELECT ... FROM table WHERE ..."""
    def col(i):
        return header[i] if 0 <= i < len(header) else f"<c{i}>"
    sel = col(q["sel"])
    if q["agg"]:
        sel = f"{AGG_OPS[q['agg']]}({sel})"
    sql = f"SELECT {sel} FROM table"
    if q["conds"]:
        sql += " WHERE " + " AND ".join(
            f"{col(c)} {COND_OPS[o]} '{v}'" for c, o, v in q["conds"])
    return sql


# --------------------------------------------- single question (front end)
def translate(model, sp, question, header, decoding="beam", device="cpu"):
    """question + list of column names -> (readable SQL or None, raw model output string)"""
    if len(header) > MAX_COLS:
        raise ValueError(f"at most {MAX_COLS} columns are supported")
    ids = sp.encode(encode_source(question, header)) + [EOS_ID]
    src = torch.tensor([ids], device=device)
    out = beam_search(model, src) if decoding == "beam" else greedy_decode(model, src)[0]
    raw = sp.decode(out)
    q = ids_to_query(sp, out, n_cols=len(header))
    return (to_readable_sql(q, header) if q else None), raw
