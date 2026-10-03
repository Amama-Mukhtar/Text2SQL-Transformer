"""Scaled dot-product attention and multi-head attention (paper, section 3.2)."""
import math

import torch
import torch.nn as nn


def attention(Q, K, V, mask=None):
    """softmax(Q K^T / sqrt(d_k)) V

    mask: True/1 = position can be seen, False/0 = hidden. It only has to be
    broadcastable to (B, heads, Lq, Lk). Returns (output, attention weights).
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        # hidden positions get -inf so softmax turns them into exactly 0
        scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, V), weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=256, h=4, dropout=0.1):
        super().__init__()
        assert d_model % h == 0, "d_model must be divisible by number of heads"
        self.h = h
        self.d_k = d_model // h  # 256 / 4 = 64
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.last_w = None  # kept so we can plot attention maps later

    def _split(self, x, lin):
        # (B, L, 256) -> (B, 4, L, 64): every head gets its own 64-dim slice
        B = x.size(0)
        return lin(x).view(B, -1, self.h, self.d_k).transpose(1, 2)

    def forward(self, q, k, v, mask=None):
        B = q.size(0)
        Q = self._split(q, self.W_q)
        K = self._split(k, self.W_k)
        V = self._split(v, self.W_v)
        out, w = attention(Q, K, V, mask)
        # glue the heads back together: (B, 4, L, 64) -> (B, L, 256)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.h * self.d_k)
        self.last_w = w.detach()
        return self.drop(self.W_o(out))
