import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = (Path(__file__).resolve().parent / ".." / ".." / "output" / "bert-finetuned-negative-positive").resolve()

if not MODEL_DIR.exists():
    raise FileNotFoundError(
        f"Local model directory not found: {MODEL_DIR}. "
        "Please train/export the model to output/bert-finetuned-negative-positive first."
    )

class PredictModel:
    def __init__(self, device, model_dir=MODEL_DIR):
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        # Force label names (prevents LABEL_0/LABEL_1 on Gradio)
        self.model.config.id2label = {0: 'NEGATIVE', 1: 'POSITIVE'}
        self.model.config.label2id = {'NEGATIVE': 0, 'POSITIVE': 1}
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    def load(self):
        #TODO: load model if needed (currently loaded in __init__)
        pass

    def predict(self,text: str):
        if text is None or not str(text).strip():
            return {"NEGATIVE": 0.0, "POSITIVE": 0.0}
        inputs = self.tokenizer(str(text), return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu().tolist()
        # Trả về dict để Gradio hiển thị dạng label->score
        return {str(self.model.config.id2label.get(i, i)): float(p) for i, p in enumerate(probs)}
    
    def predict_label(self,text: str):
        scores = self.predict(text)
        label = max(scores, key=scores.get)  # Lấy label có score cao nhất
        return label, scores
    
