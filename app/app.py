"""Task 6: small Gradio front end that calls OUR trained model.
Run from the repo root:   CKPT=checkpoints/best.pt python app/app.py
(needs sql_sp.model in the repo root and:  pip install gradio)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import gradio as gr  # noqa: E402

from common import device, load_model, load_sp  # noqa: E402
from decode import translate  # noqa: E402

CKPT = os.environ.get("CKPT", "checkpoints/best.pt")
sp = load_sp()
model = load_model(CKPT, sp)


def run(question, columns, decoding):
    header = [c.strip() for c in columns.split(",") if c.strip()]
    if not question.strip() or not header:
        return "Please type a question and at least one column name.", ""
    try:
        sql, raw = translate(model, sp, question, header, decoding, device)
    except ValueError as e:
        return str(e), ""
    return sql or "(the model output could not be parsed into SQL)", raw


demo = gr.Interface(
    fn=run,
    inputs=[
        gr.Textbox(label="Question", placeholder="What is Terrence Ross' nationality?"),
        gr.Textbox(label="Column names (comma separated)",
                   placeholder="Player, No., Nationality, Position, Years in Toronto, School/Club Team"),
        gr.Radio(["beam", "greedy"], value="beam", label="Decoding"),
    ],
    outputs=[gr.Textbox(label="Generated SQL"), gr.Textbox(label="Raw model output")],
    title="Text-to-SQL (Transformer built from scratch)",
    description="Values come out lower-cased because the model reads lower-cased text.",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch(share=True)
