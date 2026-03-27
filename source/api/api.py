from __future__ import annotations

from typing import Callable, Dict, Optional

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Positive Negative Comment Classification API")

prediction_fn = None

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Input comment/text")

class PredictResponse(BaseModel):
	label: str
	scores: dict[str, float]

@app.get("/health")
def health():
	return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
	if prediction_fn is None:
		raise RuntimeError("prediction_fn is not initialized")
	scores = prediction_fn(req.text)
	label = "POSITIVE" if scores.get("POSITIVE", 0.0) >= scores.get("NEGATIVE", 0.0) else "NEGATIVE"
	return PredictResponse(label=label, scores=scores)
    
def api_run(
	predict_fn = None,
) -> None:
	global prediction_fn
	prediction_fn = predict_fn
	uvicorn.run(app, host="0.0.0.0", port=8000)
	
def get_app() -> FastAPI:
    return app