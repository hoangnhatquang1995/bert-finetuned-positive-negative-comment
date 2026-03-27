from . import predict

device = "cpu"
_model = predict.PredictModel(device=device)

def predict_sentiment(text: str):
    return get_model().predict(text)

def load_model():
    global _model
    if(_model is None):
        _model = predict.PredictModel(device=device)

def get_model():
    load_model()
    return _model
