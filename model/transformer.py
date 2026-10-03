"""Masks and the full encoder-decoder Transformer (paper, section 3)."""
import os
import sys

import torch
import torch.nn as nn

# the given starter files live in ../starter
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "starter"))
from embeddings import InputLayer, TokenEmbedding  # noqa: E402
from tokenizer import PAD_ID  # noqa: E402

from model.layers import DecoderLayer, EncoderLayer  # noqa: E402


def make_pad_mask(ids):
    """(B, L) -> (B, 1, 1, L). True = real token, False = <pad>."""
    return (ids != PAD_ID).unsqueeze(1).unsqueeze(2)


def make_causal_mask(T, device):
    """(1, 1, T, T) lower triangle: position t can only look at positions <= t."""
    return torch.tril(torch.ones(T, T, device=device)).bool().unsqueeze(0).unsqueeze(0)


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, h=4, N=3, d_ff=1024, dropout=0.1):
        super().__init__()
        shared = TokenEmbedding(vocab_size, d_model, PAD_ID)
        self.enc_in = InputLayer(shared, d_model, dropout=dropout)
        self.dec_in = InputLayer(shared, d_model, dropout=dropout)  # same embedding weights
        self.enc_layers = nn.ModuleList([EncoderLayer(d_model, h, d_ff, dropout) for _ in range(N)])
        self.dec_layers = nn.ModuleList([DecoderLayer(d_model, h, d_ff, dropout) for _ in range(N)])
        self.out = nn.Linear(d_model, vocab_size, bias=False)
        self.out.weight = shared.emb.weight  # output projection = embedding matrix (same tensor)

        # Xavier init like the paper's reference code, then put the <pad> row back to zero
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        with torch.no_grad():
            shared.emb.weight[PAD_ID].zero_()

    def encode(self, src):
        src_mask = make_pad_mask(src)
        x = self.enc_in(src)
        for layer in self.enc_layers:
            x = layer(x, src_mask)
        return x, src_mask

    def decode(self, tgt_in, enc_out, src_mask):
        T = tgt_in.size(1)
        tgt_mask = make_pad_mask(tgt_in) & make_causal_mask(T, tgt_in.device)
        y = self.dec_in(tgt_in)
        for layer in self.dec_layers:
            y = layer(y, enc_out, tgt_mask, src_mask)
        return self.out(y)  # (B, T, vocab) logits

    def forward(self, src, tgt_in):
        enc_out, src_mask = self.encode(src)
        return self.decode(tgt_in, enc_out, src_mask)
