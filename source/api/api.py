from fastapi import FastAPI
from pydantic import BaseModel, Field


app = FastAPI(title="Positive Negative Comment Classification API")

predictor: SentimentPredictor | None = None

class PredictRequest(BaseModel):
	text: str = Field(..., min_length=1, description="Input comment/text")

class PredictResponse(BaseModel):
	label: str
	scores: dict[str, float]

@app.on_event("startup")
def _startup() -> None:
    #TODO:
	pass

@app.get("/health")
def health():
	return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
	return {
		"label": "POSITIVE",  #TODO: replace with actual prediction
		"scores": {"NEGATIVE": 0.1, "POSITIVE": 0.9}  #TODO: replace with actual scores
    }
    