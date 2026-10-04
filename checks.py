"""Correctness checks from section 3.2 (causal mask, padding mask, rows sum to 1, weight sharing).
Usage: python checks.py > results/checks.txt   (add --ckpt path/to/best.pt to test a trained model)
"""
import argparse

import torch

from common import device, load_model, load_sp
from dataset import make_loader
from model.transformer import Transformer
from tokenizer import PAD_ID

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=None)
args = ap.parse_args()

sp = load_sp()
V = sp.get_piece_size()
model = load_model(args.ckpt, sp) if args.ckpt else Transformer(V).to(device)
model.eval()  # dropout off, otherwise the comparisons are meaningless
print("trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

src, tgt = next(iter(make_loader("dev_pairs.jsonl", sp, train=False, batch_size=64)))
src, tgt = src[:4].to(device), tgt[:4].to(device)
dec_in = tgt[:, :-1]

with torch.no_grad():
    base = model(src, dec_in)

    d2 = dec_in.clone()
    d2[:, -1] = (d2[:, -1] + 7) % V  # change only the last decoder input token
    out2 = model(src, d2)
    print("causal mask ok:", torch.allclose(base[:, :-1], out2[:, :-1], atol=1e-5))

    extra = torch.full((src.size(0), 5), PAD_ID, device=device)  # 5 extra <pad> on the source
    out3 = model(torch.cat([src, extra], dim=1), dec_in)
    print("padding mask ok:", torch.allclose(base, out3, atol=1e-5))

    for name, w in [("decoder cross-attn", model.dec_layers[-1].cross_attn.last_w),
                    ("encoder self-attn", model.enc_layers[-1].attn.last_w)]:
        s = w.sum(-1)
        print(f"{name} rows sum to 1:", torch.allclose(s, torch.ones_like(s), atol=1e-5))

print("weight sharing ok:", model.out.weight is model.enc_in.tok.emb.weight
      and model.enc_in.tok is model.dec_in.tok)
