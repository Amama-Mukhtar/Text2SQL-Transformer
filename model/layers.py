"""Feed-forward, encoder layer and decoder layer (paper, sections 3.1 - 3.3)."""
import torch
import torch.nn as nn

from model.attention import MultiHeadAttention


class FeedForward(nn.Module):
    """FFN(x) = max(0, x W1 + b1) W2 + b2"""

    def __init__(self, d_model=256, d_ff=1024, dropout=0.1):
        super().__init__()
        self.l1 = nn.Linear(d_model, d_ff)
        self.l2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.l2(torch.relu(self.l1(x))))


class EncoderLayer(nn.Module):
    # post-norm like the paper: LayerNorm(x + Sublayer(x))
    def __init__(self, d_model=256, h=4, d_ff=1024, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, h, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.n1 = nn.LayerNorm(d_model)
        self.n2 = nn.LayerNorm(d_model)

    def forward(self, x, src_mask):
        x = self.n1(x + self.attn(x, x, x, src_mask))
        x = self.n2(x + self.ff(x))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model=256, h=4, d_ff=1024, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, h, dropout)
        self.cross_attn = MultiHeadAttention(d_model, h, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.n1 = nn.LayerNorm(d_model)
        self.n2 = nn.LayerNorm(d_model)
        self.n3 = nn.LayerNorm(d_model)

    def forward(self, y, enc_out, tgt_mask, src_mask):
        y = self.n1(y + self.self_attn(y, y, y, tgt_mask))               # masked self-attention
        y = self.n2(y + self.cross_attn(y, enc_out, enc_out, src_mask))  # look at the encoder
        y = self.n3(y + self.ff(y))
        return y
