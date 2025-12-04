from __future__ import annotations
from pydantic import BaseModel


class SentimentRequest(BaseModel):
    text: str


class SentimentResponse(BaseModel):
    label: int
    is_fake: bool
    score: float | None = None