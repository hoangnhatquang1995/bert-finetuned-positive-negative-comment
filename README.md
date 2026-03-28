# Positive Negative Vietnamese Comment Prediction

## Problem

Vietnamese sentiment classification for user comments: predict **POSITIVE** vs **NEGATIVE**.

## Approach

- Fine-tune a Transformer model (**PhoBERT**) on a labeled sentiment dataset
- Support 2 training/export modes:
	- **Full fine-tune** (export full model) to `output/bert-finetuned-negative-positive`
	- **LoRA / PEFT** (export adapter) to `output/phobert_lora_sentiment`
- Serve one app with:
	- **FastAPI** JSON API (`/health`, `/predict`)
	- **Gradio UI** mounted at `/ui`

## Result

- Test split (Full fine-tune PhoBERT, max_length=256):
	- Accuracy: **94.56%**
	- F1 (overall / POSITIVE): **0.9549**
	- F1 (NEGATIVE): **0.9314**
- Test split (PhoBERT + LoRA adapter):
	- Accuracy: **92.97%**
	- F1 (overall): **0.9428**
- Full report + confusion matrix are printed and exported to `output/metrics.json` by:
	- `source/finetune/finetuning.ipynb`

## Demo

- UI: `http://127.0.0.1:8000/ui`
- Health: `http://127.0.0.1:8000/health`

API example:

```powershell
python -c "import requests; print(requests.post('http://127.0.0.1:8000/predict', json={'text':'Quả táo này rất ngon'}).text)"
```

Example response:

```json
{
	"label": "POSITIVE",
	"scores": {
		"NEGATIVE": 0.02,
		"POSITIVE": 0.98
	}
}
```

---

Vietnamese sentiment classification (POSITIVE/NEGATIVE) fine-tuned from PhoBERT and deployed as:

- **Gradio UI** for interactive demo
- **FastAPI** for a simple JSON prediction API

## Features

- Fine-tune a Transformer model on a positive/negative comment dataset
- Support inference with **LoRA / PEFT** adapter (PhoBERT + LoRA)
- Shared prediction module (loaded once, reused for UI + API)
- Simple endpoints for integration: `/health`, `/predict`

## Project Structure

```
datasets/                 # training data
output/                   # exported model (local)
source/
	prediction/             # prediction package (predict_sentiment exposed via __init__)
	ui/                     # Gradio UI
	api/                    # FastAPI service
	finetune/               # notebooks for training
	app.py                  # entrypoint (run UI or API)
```

## Requirements

- Python **3.12**
- (Optional) NVIDIA GPU + CUDA for faster training/inference

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Train / Export Model

This repo supports both **full fine-tune** and **LoRA**.

### Option A — Full fine-tune

Run:

- `source/finetune/finetuning.ipynb`

After training, export the model to:

```
output/bert-finetuned-negative-positive
```

### Option B — LoRA (PEFT)

Run:

- `source/finetune/finetuning_lora.ipynb`

After training, export the LoRA adapter to:

```
output/phobert_lora_sentiment
```

> By default, inference in this app uses the LoRA adapter.

## Run App (API + UI)

This starts **one FastAPI server** and mounts the **Gradio UI** at `/ui`.

```powershell
python -m source.app
```

## Docker

### Build

```bash
docker build -t sentiment-app .
```

### Run

```bash
docker run --rm -p 8000:8000 sentiment-app
```

### Use model as a volume (optional)

If you don't want to bake the model into the image, you can mount `output/`:

```bash
docker run --rm -p 8000:8000 -v "${PWD}/output:/app/output" sentiment-app
```

## Notes

- Prediction entrypoint is exposed via `source/prediction/__init__.py` as `predict_sentiment(text)`.
- Default model mode is **LoRA** (see `PredictModel(used_lora=True)` in `source/prediction/__init__.py`).
- If your exported folder is missing, the app/API will raise `FileNotFoundError`.

