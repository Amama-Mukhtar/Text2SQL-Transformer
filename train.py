"""Task 3: train the Transformer from random init.

Colab example:
    python train.py --ckpt-dir /content/drive/MyDrive/genai_a2
If the session dies, run the same command again and it resumes from last.pt.
"""
import argparse
import json
import os
import time

import torch
import torch.nn as nn

from common import device, load_sp
from dataset import make_loader
from model.transformer import Transformer
from tokenizer import PAD_ID

D_MODEL, WARMUP = 256, 4000


def noam_lr(step, d_model=D_MODEL, warmup=WARMUP):
    """paper eq. 3: d_model^-0.5 * min(step^-0.5, step * warmup^-1.5)"""
    step = max(step, 1)  # avoid 0 ** -0.5
    return d_model ** -0.5 * min(step ** -0.5, step * warmup ** -1.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--max-steps-per-epoch", type=int, default=0, help="only for quick smoke tests")
    args = ap.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)
    torch.manual_seed(42)

    sp = load_sp()
    V = sp.get_piece_size()
    model = Transformer(V).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("trainable params:", n_params)

    train_dl = make_loader("train_pairs.jsonl", sp, train=True, batch_size=args.batch_size)
    dev_dl = make_loader("dev_pairs.jsonl", sp, train=False, batch_size=args.batch_size)

    # label smoothing 0.1, <pad> is ignored
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=0.1)
    opt = torch.optim.Adam(model.parameters(), lr=0.0, betas=(0.9, 0.98), eps=1e-9)

    last_path = os.path.join(args.ckpt_dir, "last.pt")
    best_path = os.path.join(args.ckpt_dir, "best.pt")
    start_epoch, step, best_dev, history, train_secs = 0, 0, float("inf"), [], 0.0

    if os.path.exists(last_path):  # resume after a disconnect
        ck = torch.load(last_path, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_epoch, step = ck["epoch"], ck["step"]
        best_dev, history, train_secs = ck["best_dev"], ck["history"], ck["train_secs"]
        print(f"resumed from epoch {start_epoch}, step {step}")

    def run_loss(src, tgt):
        # teacher forcing: decoder input = tgt[:, :-1], target = tgt[:, 1:]
        logits = model(src, tgt[:, :-1])
        return criterion(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))

    @torch.no_grad()
    def dev_loss():
        model.eval()
        tot, n = 0.0, 0
        for i, (src, tgt) in enumerate(dev_dl):
            if args.max_steps_per_epoch and i >= args.max_steps_per_epoch:
                break
            tot += run_loss(src.to(device), tgt.to(device)).item()
            n += 1
        return tot / n

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for i, (src, tgt) in enumerate(train_dl):
            if args.max_steps_per_epoch and i >= args.max_steps_per_epoch:
                break
            src, tgt = src.to(device), tgt.to(device)
            step += 1
            lr = noam_lr(step)
            for g in opt.param_groups:
                g["lr"] = lr
            loss = run_loss(src, tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            n += 1

        train_secs += time.time() - t0
        tr, dv = tot / n, dev_loss()
        history.append({"epoch": epoch + 1, "train_loss": tr, "dev_loss": dv, "lr": lr})
        print(f"epoch {epoch+1:2d} | train {tr:.4f} | dev {dv:.4f} | lr {lr:.6f} | {time.time()-t0:.0f}s")

        if dv < best_dev:  # keep the checkpoint with the lowest dev loss
            best_dev = dv
            torch.save({"model": model.state_dict(), "epoch": epoch + 1, "dev_loss": dv}, best_path)
            print("   ^ best so far, saved")

        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch + 1,
                    "step": step, "best_dev": best_dev, "history": history,
                    "train_secs": train_secs}, last_path)
        json.dump(history, open("results/history.json", "w"), indent=1)

    best_ep = min(history, key=lambda h: h["dev_loss"])["epoch"]
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    summary = {"trainable_params": n_params, "epochs_trained": len(history), "best_epoch": best_ep,
               "best_dev_loss": best_dev, "train_minutes": round(train_secs / 60, 1), "gpu": gpu}
    json.dump(summary, open("results/train_summary.json", "w"), indent=1)
    print(summary)


if __name__ == "__main__":
    main()
