
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)

tokenizer = AutoTokenizer.from_pretrained('local_models/')
model = AutoModelForSequenceClassification.from_pretrained('local_models/')

# return_all_scores=True → вероятности по всем классам тональности
clf = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer
)