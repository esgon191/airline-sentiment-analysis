from __future__ import annotations
from fastapi import FastAPI, HTTPException

from src.api.schemas import SentimentRequest
from src.models.infer import SentimentInferenceModel

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)


app = FastAPI(
    title="Airline Sentiment Analysis API",
    description="API предсказания тональности отзывов positive / neutral / negative",
    version="0.2.0",
)

pipe : pipeline | None = None

@app.on_event("startup")
def load_model() -> None:
    global pipe
    try:
        tokenizer = AutoTokenizer.from_pretrained('local_models/')
        model = AutoModelForSequenceClassification.from_pretrained('local_models/')

        # return_all_scores=True → вероятности по всем классам тональности
        clf = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer
        )
        
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to load model: {exc}")
        pipe = None


@app.get("/health")
def health() -> dict:
    status = "200 OK" if pipe is not None else "model_not_loaded"
    return {"status": status}


@app.post("/predict")
def predict(request: SentimentRequest):
    pipe = getattr(app.state, "pipe", None)

    if pipe is None:
        # модель не загружена – значит сервис недоступен
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Cannot perform prediction.",
        )
    try:
        result = pipe(request.text)
        return result
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc