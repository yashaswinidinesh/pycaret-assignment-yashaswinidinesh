
import gradio as gr
import pandas as pd
from pycaret.classification import load_model, predict_model

model = load_model("bank_marketing_classifier")

def predict_csv(file):
    df = pd.read_csv(file.name)
    out = predict_model(model, data=df)
    return out.to_csv(index=False)

demo = gr.Interface(fn=predict_csv,
                    inputs=gr.File(label="Upload CSV with columns used in training"),
                    outputs=gr.File(label="Predictions CSV"),
                    title="PyCaret Classifier — CSV Batch Scoring",
                    allow_flagging="never")

if __name__ == "__main__":
    demo.launch()
