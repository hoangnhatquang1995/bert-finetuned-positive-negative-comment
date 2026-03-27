import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = (Path(__file__).resolve().parent / ".." / ".." / "output" / "bert-finetuned-negative-positive").resolve()

if not MODEL_DIR.exists():
    raise FileNotFoundError(
        f"Local model directory not found: {MODEL_DIR}. "
        "Please train/export the model to output/bert-finetuned-negative-positive first."
    )

tokenizer_deploy = AutoTokenizer.from_pretrained(str(MODEL_DIR))
model_deploy = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
# Force label names (prevents LABEL_0/LABEL_1 on Gradio)
model_deploy.config.id2label = {0: 'NEGATIVE', 1: 'POSITIVE'}
model_deploy.config.label2id = {'NEGATIVE': 0, 'POSITIVE': 1}

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
model_deploy.to(device)
model_deploy.eval()

id2label = model_deploy.config.id2label

def predict_sentiment(text: str):
    if text is None or not str(text).strip():
        return {"NEGATIVE": 0.0, "POSITIVE": 0.0}
    inputs = tokenizer_deploy(str(text), return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model_deploy(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu().tolist()
    # Trả về dict để Gradio hiển thị dạng label->score
    return {str(id2label.get(i, i)): float(p) for i, p in enumerate(probs)}

import gradio as gr

demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=3, placeholder="Nhập bình luận..."),
    outputs=gr.Label(num_top_classes=2),
    title="Sentiment (PhoBERT fine-tuned)",
    description="Demo deploy nhanh trên Colab: POSITIVE/NEGATIVE",
    examples=[
        ["Quả táo này rất ngon"] ,
        ["Quả táo này rất dở"] ,
    ],
 )

# share=True để lấy public URL khi chạy trên Colab
demo.launch(share=True, debug=False)