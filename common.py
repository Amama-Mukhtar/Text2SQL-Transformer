"""Small helpers shared by the scripts. Run everything from the repo root."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "starter"))

import sentencepiece as spm  # noqa: E402
import torch  # noqa: E402

from model.transformer import Transformer  # noqa: E402

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_sp(path="sql_sp.model"):
    return spm.SentencePieceProcessor(model_file=path)


def load_model(ckpt_path, sp, dev=None):
    dev = dev or device
    model = Transformer(sp.get_piece_size()).to(dev)
    ck = torch.load(ckpt_path, map_location=dev)
    model.load_state_dict(ck["model"])
    model.eval()
    return model
