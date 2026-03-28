import torch
from pathlib import Path
from typing import Dict, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel,PeftConfig

FINETUNE_MODEL_DIR = (Path(__file__).resolve().parent / ".." / ".." / "output" / "bert-finetuned-negative-positive").resolve()
LORA_MODEL_DIR = (Path(__file__).resolve().parent / ".." / ".." / "output" / "phobert_lora_sentiment").resolve()

class PredictModel:
    id2label = {0: 'NEGATIVE', 1: 'POSITIVE'}
    label2id = {'NEGATIVE': 0, 'POSITIVE': 1}

    def __init__(self, used_lora=True, device = "cpu"):
        if used_lora:
            if not LORA_MODEL_DIR.exists():
                raise FileNotFoundError(
                    f"Local LoRA adapter directory not found: {LORA_MODEL_DIR}. "
                    "Please train/export the adapter to output/phobert_lora_sentiment first."
                )
            config = PeftConfig.from_pretrained(str(LORA_MODEL_DIR))
            base_model_name = config.base_model_name_or_path
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            base_model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=2)
            self.model = PeftModel.from_pretrained(base_model, str(LORA_MODEL_DIR))
            setattr(self.model.config, "id2label", self.id2label)
            setattr(self.model.config, "label2id", self.label2id)
        else:
            if not FINETUNE_MODEL_DIR.exists():
                raise FileNotFoundError(
                    f"Local fine-tuned model directory not found: {FINETUNE_MODEL_DIR}. "
                    "Please train/export the model to output/bert-finetuned-negative-positive first."
                )
            self.tokenizer = AutoTokenizer.from_pretrained(str(FINETUNE_MODEL_DIR))
            self.model = AutoModelForSequenceClassification.from_pretrained(str(FINETUNE_MODEL_DIR))
            setattr(self.model.config, "id2label", self.id2label)
            setattr(self.model.config, "label2id", self.label2id)
        self.device = device
        self.model.to(self.device)
        self.model.eval()


    def predict(self, text: str) -> Dict[str, float]:
        if text is None or not str(text).strip():
            return {"NEGATIVE": 0.0, "POSITIVE": 0.0}
        inputs = self.tokenizer(str(text), return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu().tolist()
        # Trả về dict để Gradio hiển thị dạng label->score
        id2label = getattr(self.model.config, "id2label", self.id2label)
        if not isinstance(id2label, dict):
            id2label = self.id2label
        return {str(id2label.get(i, i)): float(p) for i, p in enumerate(probs)}
    
    def predict_label(self, text: str) -> Tuple[str, Dict[str, float]]:
        scores = self.predict(text)
        label = max(scores, key=lambda k: scores[k])  # Lấy label có score cao nhất
        return label, scores
    
