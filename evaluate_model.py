"""Task 5: predictions, official evaluator, component accuracy, attention map, samples.

    python evaluate_model.py dev  --ckpt checkpoints/best.pt            # greedy + beam + everything on dev
    python evaluate_model.py test --ckpt checkpoints/best.pt --decode beam   # run ONCE at the very end
Add --limit 200 for a quick smoke test (then delete results/ files and run the real thing).
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from common import device, load_model, load_sp  # noqa: E402
from data_prep import load_split  # noqa: E402
from decode import beam_search, greedy_decode, ids_to_query, parse_query, to_readable_sql  # noqa: E402
from tokenizer import BOS_ID, EOS_ID, PAD_ID, read_pairs  # noqa: E402

WIKISQL = "WikiSQL"
RES = "results"


# --------------------------------------------------------------- predictions
def predict(model, sp, split, decoding, limit=0, bs=64):
    """Returns one list of token ids per example, in the same order as {split}.jsonl."""
    pairs = read_pairs(f"{split}_pairs.jsonl")
    if limit:
        pairs = pairs[:limit]
    srcs = [sp.encode(p["src"]) + [EOS_ID] for p in pairs]
    out = []
    if decoding == "greedy":
        for i in tqdm(range(0, len(srcs), bs), desc=f"{split} greedy"):
            chunk = srcs[i:i + bs]
            S = max(map(len, chunk))
            batch = torch.full((len(chunk), S), PAD_ID, dtype=torch.long)
            for j, s in enumerate(chunk):
                batch[j, :len(s)] = torch.tensor(s)
            out += greedy_decode(model, batch.to(device))
    else:
        for s in tqdm(srcs, desc=f"{split} beam(4)"):
            out.append(beam_search(model, torch.tensor([s], device=device), beam=4))
    return out


def to_queries(sp, id_lists, examples, tables):
    qs = []
    for ids, ex in zip(id_lists, examples):
        n_cols = len(tables[ex["table_id"]]["header"])
        qs.append(ids_to_query(sp, ids, n_cols=n_cols))
    return qs


def write_predictions(path, queries):
    with open(path, "w", encoding="utf-8") as f:
        for q in queries:  # one line per example, never skip one
            f.write(json.dumps({"query": q} if q is not None else {"error": "parse"}) + "\n")


def official_eval(split, pred_path):
    r = subprocess.run([sys.executable, "evaluate.py", f"data/{split}.jsonl", f"data/{split}.db",
                        os.path.abspath(pred_path)], cwd=WIKISQL, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-2000:])
    d = json.loads(r.stdout[r.stdout.index("{"):])
    return round(100 * d["lf_accuracy"], 2), round(100 * d["ex_accuracy"], 2)


# ---------------------------------------------------------- component accuracy
def cond_set(conds):
    return {(int(c), int(o), str(v).lower()) for c, o, v in conds}


def component_accuracy(queries, examples):
    n = len(queries)
    sel = agg = where = 0
    for q, ex in zip(queries, examples):
        g = ex["sql"]
        if q is None:
            continue
        sel += q["sel"] == g["sel"]
        agg += q["agg"] == g["agg"]
        where += cond_set(q["conds"]) == cond_set(g["conds"])
    return {"sel": round(100 * sel / n, 2), "agg": round(100 * agg / n, 2), "where": round(100 * where / n, 2)}


def failure_type(q, g):
    if q is None:
        return "parse failure"
    if q["sel"] != g["sel"]:
        return "wrong select column"
    if q["agg"] != g["agg"]:
        return "wrong aggregation"
    pc, gc = cond_set(q["conds"]), cond_set(g["conds"])
    if pc == gc:
        return "correct"
    if len(pc) < len(gc):
        return "missing condition"
    if len(pc) > len(gc):
        return "extra condition"
    if {c for c, _, _ in pc} != {c for c, _, _ in gc}:
        return "wrong condition column"
    return "wrong value or operator"


# ----------------------------------------------------------------- samples.md
def write_samples(queries, examples, tables, decoding, path=f"{RES}/samples.md"):
    right, wrong, seen = [], [], set()
    rows = [(i, failure_type(q, ex["sql"])) for i, (q, ex) in enumerate(zip(queries, examples))]
    right = [i for i, t in rows if t == "correct"][:5]
    bad = [(i, t) for i, t in rows if t != "correct"]
    for i, t in bad:  # prefer different failure types
        if t not in seen and len(wrong) < 5:
            wrong.append((i, t))
            seen.add(t)
    for i, t in bad:
        if len(wrong) < 5 and (i, t) not in wrong:
            wrong.append((i, t))

    def block(i, tag):
        ex = examples[i]
        header = tables[ex["table_id"]]["header"]
        q = queries[i]
        return (f"**Question:** {ex['question']}\n\n"
                f"- Gold: `{to_readable_sql(ex['sql'], header)}`\n"
                f"- Ours ({decoding}): `{to_readable_sql(q, header) if q else 'PARSE FAILURE'}`\n"
                f"- Verdict: {tag}\n")

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Qualitative samples (dev, {decoding})\n\nValues are lower-cased because the model works on lower-cased text.\n\n## Correct\n\n")
        for k, i in enumerate(right, 1):
            f.write(f"### {k}\n" + block(i, "correct") + "\n")
        f.write("## Wrong\n\n")
        for k, (i, t) in enumerate(wrong, 1):
            f.write(f"### {k}\n" + block(i, t) + "\n")


# -------------------------------------------------------------- attention map
def source_columns(pieces):
    """for every source piece, which column (0,1,2..) it belongs to (None before the first <cK>)."""
    col, out = None, []
    for p in pieces:
        m = re.fullmatch(r"<c(\d+)>", p)
        if m:
            col = int(m.group(1))
        out.append(col)
    return out


def cross_attention(model, sp, pair, gen_ids):
    src_ids = sp.encode(pair["src"]) + [EOS_ID]
    gen = gen_ids + [EOS_ID]
    dec_in = [BOS_ID] + gen[:-1]  # row i of the map = the step that produced gen[i]
    with torch.no_grad():
        model(torch.tensor([src_ids], device=device), torch.tensor([dec_in], device=device))
    w = model.dec_layers[-1].cross_attn.last_w[0].mean(0).cpu().numpy()  # average over heads: (T, S)
    return w, [sp.id_to_piece(i) for i in src_ids], [sp.id_to_piece(i) for i in gen]


def attention_report(model, sp, pairs, id_lists, idx, n_stat=200):
    w, xs, ys = cross_attention(model, sp, pairs[idx], id_lists[idx])
    fig, ax = plt.subplots(figsize=(max(8, 0.28 * len(xs)), max(3, 0.35 * len(ys))))
    im = ax.imshow(w, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels(xs, rotation=90, fontsize=7)
    ax.set_yticks(range(len(ys)))
    ax.set_yticklabels(ys, fontsize=8)
    ax.set_xlabel("source tokens")
    ax.set_ylabel("generated tokens")
    ax.set_title(f"Decoder cross-attention, last layer (mean over heads), dev example {idx}")
    fig.colorbar(im)
    fig.savefig(f"{RES}/attention_map.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    lines = [f"dev example {idx}: {pairs[idx]['src']}", ""]
    cols = source_columns(xs)
    for t, y in enumerate(ys):
        if re.fullmatch(r"<c\d+>", y):
            pos = int(w[t].argmax())
            ok = cols[pos] == int(y[2:-1])
            lines.append(f"generated {y} -> most attended source token: {xs[pos]} (belongs to column {cols[pos]}) match={ok}")
    # same check over the first n_stat examples: does a generated <cK> attend inside column K's span?
    hit = tot = 0
    for j in range(min(n_stat, len(pairs))):
        w_j, xs_j, ys_j = cross_attention(model, sp, pairs[j], id_lists[j])
        cj = source_columns(xs_j)
        for t, y in enumerate(ys_j):
            if re.fullmatch(r"<c\d+>", y):
                tot += 1
                hit += cj[int(w_j[t].argmax())] == int(y[2:-1])
    lines += ["", f"over the first {min(n_stat, len(pairs))} dev examples: {hit}/{tot} generated column tokens "
                  f"({100 * hit / max(tot, 1):.1f}%) have their strongest attention inside the matching source column"]
    open(f"{RES}/attention_check.txt", "w").write("\n".join(lines))
    print("\n".join(lines))


# ------------------------------------------------------------------ tables
def load_metrics():
    p = f"{RES}/metrics.json"
    return json.load(open(p)) if os.path.exists(p) else {}


def save_metrics(m):
    json.dump(m, open(f"{RES}/metrics.json", "w"), indent=1)
    make_tables(m)


def make_tables(m):
    L = []
    if os.path.exists(f"{RES}/table1.json"):
        t = json.load(open(f"{RES}/table1.json"))
        L += ["## Table 1 - Data (lengths include <s> and </s>)", "",
              "| | Train | Dev | Test |", "|---|---|---|---|",
              "| Pairs | " + " | ".join(str(t[s]["pairs"]) for s in ["train", "dev", "test"]) + " |",
              "| Mean / max source length | " + " | ".join(f"{t[s]['src_mean']} / {t[s]['src_max']}" for s in ["train", "dev", "test"]) + " |",
              "| Mean / max target length | " + " | ".join(f"{t[s]['tgt_mean']} / {t[s]['tgt_max']}" for s in ["train", "dev", "test"]) + " |",
              "| Pairs dropped as too long | " + " | ".join(str(t[s]["dropped"]) for s in ["train", "dev", "test"]) + " |", ""]
    if os.path.exists(f"{RES}/train_summary.json"):
        s = json.load(open(f"{RES}/train_summary.json"))
        L += ["## Table 2 - Model and training", "",
              f"- Trainable parameters: {s['trainable_params']:,}",
              f"- Epochs trained / best epoch: {s['epochs_trained']} / {s['best_epoch']}",
              f"- Best dev loss: {s['best_dev_loss']:.4f}",
              f"- Training time and GPU: {s['train_minutes']} min on {s['gpu']}", ""]
    L += ["## Table 3 - Official metrics", "", "| Split | Decoding | Logical form (%) | Execution (%) | Parse failures (%) |", "|---|---|---|---|---|"]
    for key in ["dev_greedy", "dev_beam", "test_greedy", "test_beam"]:
        if key in m:
            r = m[key]
            sp_, dec = key.split("_")
            L.append(f"| {sp_.capitalize()} | {dec} | {r['lf']} | {r['ex']} | {r['parse_fail']} |")
    L += [""]
    for k, r in m.items():
        if k.startswith("component_"):
            L += [f"## Table 4 - Component accuracy (dev, {k.split('_')[1]})", "",
                  f"- sel column correct: {r['sel']}%", f"- agg correct: {r['agg']}%", f"- WHERE clause correct: {r['where']}%", ""]
    if "gold_roundtrip" in m:
        g = m["gold_roundtrip"]
        L += ["## Gold round-trip (dev gold targets -> our parser -> official evaluator)", "",
              f"- Logical form {g['lf']}%, execution {g['ex']}%, parse failures {g['parse_fail']}%", ""]
    open(f"{RES}/tables.md", "w").write("\n".join(L))


# -------------------------------------------------------------------- main
def run_split(model, sp, split, decoding, limit, m):
    examples, tables = load_split(split)
    if limit:
        examples = examples[:limit]
    t0 = time.time()
    ids = predict(model, sp, split, decoding, limit)
    qs = to_queries(sp, ids, examples, tables)
    path = f"{RES}/{split}_{decoding}.jsonl"
    write_predictions(path, qs)
    lf, ex = official_eval(split, path)
    pf = round(100 * sum(q is None for q in qs) / len(qs), 2)
    m[f"{split}_{decoding}"] = {"lf": lf, "ex": ex, "parse_fail": pf, "n": len(qs)}
    print(f"[{split} {decoding}] logical form {lf}% | execution {ex}% | parse failures {pf}% | {time.time()-t0:.0f}s")
    return ids, qs, examples, tables


def gold_roundtrip(sp, limit, m):
    examples, tables = load_split("dev")
    pairs = read_pairs("dev_pairs.jsonl")
    if limit:
        examples, pairs = examples[:limit], pairs[:limit]
    qs = [parse_query(p["tgt"]) for p in pairs]  # gold target text -> parser
    write_predictions(f"{RES}/dev_gold_roundtrip.jsonl", qs)
    lf, ex = official_eval("dev", f"{RES}/dev_gold_roundtrip.jsonl")
    pf = round(100 * sum(q is None for q in qs) / len(qs), 2)
    m["gold_roundtrip"] = {"lf": lf, "ex": ex, "parse_fail": pf}
    print(f"[gold round-trip] logical form {lf}% | execution {ex}% | parse failures {pf}%   (execution must be > 99%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=["dev", "test"])
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--decode", choices=["greedy", "beam"], default="beam", help="only used for test")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(RES, exist_ok=True)

    sp = load_sp()
    model = load_model(args.ckpt, sp)
    m = load_metrics()

    if args.split == "test":
        run_split(model, sp, "test", args.decode, args.limit, m)  # test set: once, final decoding choice
        save_metrics(m)
        return

    gold_roundtrip(sp, args.limit, m)
    results = {}
    for dec in ["greedy", "beam"]:
        results[dec] = run_split(model, sp, "dev", dec, args.limit, m)
        m[f"component_{dec}"] = component_accuracy(results[dec][1], results[dec][2])
        print(f"[dev {dec}] components:", m[f"component_{dec}"])
        save_metrics(m)

    best = "beam" if m["dev_beam"]["ex"] >= m["dev_greedy"]["ex"] else "greedy"
    print("better decoding on dev (by execution accuracy):", best)
    ids, qs, examples, tables = results[best]
    write_samples(qs, examples, tables, best)
    attention_report(model, sp, read_pairs("dev_pairs.jsonl")[:len(ids)], ids, idx=0)
    save_metrics(m)


if __name__ == "__main__":
    main()
