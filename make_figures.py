"""Figures + Table 1 that don't need the trained model:
PE heat-map, LR schedule, loss curves (needs results/history.json), Table 1 numbers.
Usage: python make_figures.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "starter"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import sentencepiece as spm  # noqa: E402
from embeddings import PositionalEncoding  # noqa: E402
from tokenizer import read_pairs  # noqa: E402

os.makedirs("results", exist_ok=True)


def noam_lr(step, d_model=256, warmup=4000):
    """same schedule as train.py (paper eq. 3), copied here so this script needs no model code"""
    step = max(step, 1)
    return d_model ** -0.5 * min(step ** -0.5, step * warmup ** -1.5)


# --- positional encoding heat-map (first 100 positions x 256 dims)
pe = PositionalEncoding(256).pe[0, :100, :].numpy()
plt.figure(figsize=(9, 4))
plt.imshow(pe, aspect="auto", cmap="RdBu")
plt.colorbar(label="value")
plt.xlabel("dimension (0-255)")
plt.ylabel("position (0-99)")
plt.title("Positional encoding")
plt.savefig("results/pe_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()

# --- learning-rate schedule, first 20,000 steps
steps = list(range(1, 20001))
plt.figure(figsize=(7, 3.5))
plt.plot(steps, [noam_lr(s) for s in steps])
plt.axvline(4000, color="gray", linestyle="--", label="warmup = 4000")
plt.xlabel("step")
plt.ylabel("learning rate")
plt.title("LR schedule (first 20,000 steps)")
plt.legend()
plt.savefig("results/lr_schedule.png", dpi=150, bbox_inches="tight")
plt.close()

# --- training / dev loss per epoch
if os.path.exists("results/history.json"):
    h = json.load(open("results/history.json"))
    ep = [x["epoch"] for x in h]
    plt.figure(figsize=(7, 3.5))
    plt.plot(ep, [x["train_loss"] for x in h], marker="o", label="train")
    plt.plot(ep, [x["dev_loss"] for x in h], marker="o", label="dev")
    plt.xlabel("epoch")
    plt.ylabel("loss (label smoothing 0.1)")
    plt.title("Training and dev loss")
    plt.legend()
    plt.savefig("results/loss_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
else:
    print("results/history.json not found, skipping loss curves")

# --- Table 1 (lengths include <s> and </s>, same counting as dataset.py)
sp = spm.SentencePieceProcessor(model_file="sql_sp.model")
table1 = {}
for split in ["train", "dev", "test"]:
    pairs = read_pairs(f"{split}_pairs.jsonl")
    sl = np.array([len(x) + 1 for x in sp.encode([p["src"] for p in pairs])])
    tl = np.array([len(x) + 2 for x in sp.encode([p["tgt"] for p in pairs])])
    too_long = int(((sl > 160) | (tl > 64)).sum())
    table1[split] = {"pairs": len(pairs),
                     "src_mean": round(float(sl.mean()), 1), "src_max": int(sl.max()),
                     "tgt_mean": round(float(tl.mean()), 1), "tgt_max": int(tl.max()),
                     # only train drops long pairs; dev/test keep every row for evaluation
                     "dropped": too_long if split == "train" else 0,
                     "longer_than_limit": too_long}
    print(split, table1[split])
json.dump(table1, open("results/table1.json", "w"), indent=1)
print("saved figures + results/table1.json")
