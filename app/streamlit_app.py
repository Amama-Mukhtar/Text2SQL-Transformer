"""Streamlit front end for the Text-to-SQL Transformer (same model as app/app.py).
Run from the repo root:   streamlit run app/streamlit_app.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import streamlit as st  # noqa: E402

from common import device, load_model, load_sp  # noqa: E402
from decode import translate  # noqa: E402

CKPT = os.environ.get("CKPT", "checkpoints/best.pt")


@st.cache_resource
def get_model():
    sp = load_sp()
    return sp, load_model(CKPT, sp)


sp, model = get_model()

st.title("Text-to-SQL (Transformer built from scratch)")
st.caption("Values come out lower-cased because the model reads lower-cased text.")

question = st.text_input("Question", placeholder="What is Terrence Ross' nationality?")
columns = st.text_input(
    "Column names (comma separated)",
    placeholder="Player, No., Nationality, Position, Years in Toronto, School/Club Team",
)
decoding = st.radio("Decoding", ["beam", "greedy"], horizontal=True)

if st.button("Generate SQL"):
    header = [c.strip() for c in columns.split(",") if c.strip()]
    if not question.strip() or not header:
        st.warning("Please type a question and at least one column name.")
    else:
        try:
            sql, raw = translate(model, sp, question, header, decoding, device)
        except ValueError as e:
            st.error(str(e))
        else:
            st.subheader("Generated SQL")
            st.code(sql or "(the model output could not be parsed into SQL)", language="sql")
            st.text_area("Raw model output", raw, height=100)