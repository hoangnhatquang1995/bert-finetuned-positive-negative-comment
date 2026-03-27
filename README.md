# Positive Negative Vietnamese Comment Prediction

Vietnamese sentiment classification (POSITIVE/NEGATIVE) fine-tuned from PhoBERT and deployed as:

- **Gradio UI** for interactive demo
- **FastAPI** for a simple JSON prediction API

## Features

- Fine-tune a Transformer model on a positive/negative comment dataset
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

Run the training notebook:

- `source/finetune/finetuning.ipynb`

After training, export the model to this folder:

```
output/bert-finetuned-negative-positive
```

> The UI and API will load from this folder by default.

## Run Gradio UI

```powershell
python -m source.app --ui
```

Open the printed local URL (e.g. `http://127.0.0.1:7861`).

## Run FastAPI

```powershell
python -m source.app --api --host 127.0.0.1 --port 8000
```

Health check:

```powershell
python -c "import requests; print(requests.get('http://127.0.0.1:8000/health').text)"
```

Predict:

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

## Notes

- Prediction entrypoint is exposed via `source/prediction/__init__.py` as `predict_sentiment(text)`.
- If `output/bert-finetuned-negative-positive` is missing, the app/API will raise `FileNotFoundError`.

## License

This project is for educational/portfolio purposes.