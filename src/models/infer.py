import pathlib
from typing import List, Dict

import numpy as np
from omegaconf import OmegaConf
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)


class SentimentInferenceModel:
    """
    Обёртка над BERT-моделью для инференса тональности текста.
    """

    def __init__(self, clf_pipeline, id2label: dict, label2id: dict):
        self.clf = clf_pipeline
        self.id2label = id2label
        self.label2id = label2id
        self.num_labels = len(id2label)

    @classmethod
    def from_config(cls, config_path: str = "configs/inference.yaml") -> "SentimentInferenceModel":
        """
        Ожидается, что в конфиге есть:
        model:
          path: "local_models/"   # папка с распакованной HF-моделью тональности
        """
        cfg = OmegaConf.load(config_path)
        model_dir = pathlib.Path(cfg.model.path)

        if not model_dir.exists():
            raise FileNotFoundError(f"Model dir not found: {model_dir}")

        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)

        id2label = model.config.id2label   # например: {0: "negative", 1: "neutral", 2: "positive"}
        label2id = model.config.label2id

        # return_all_scores=True → вероятности по всем классам тональности
        clf = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            return_all_scores=True,
        )

        return cls(clf_pipeline=clf, id2label=id2label, label2id=label2id)

    def _prepare_texts(self, texts: List[str]) -> List[str]:
        cleaned = []
        for t in texts:
            if t is None:
                cleaned.append("")
            else:
                cleaned.append(str(t))
        return cleaned

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Возвращает матрицу вероятностей формы (n_samples, n_classes),
        где столбцы соответствуют классам тональности в порядке id2label[0], id2label[1], ...
        """
        texts_clean = self._prepare_texts(texts)

        # List[List[{'label': 'negative', 'score': 0.72}, {...}, ...]]
        outputs = self.clf(texts_clean)

        labels_sorted = sorted(self.label2id.items(), key=lambda x: x[1])  # [(label_str, id), ...]
        num_classes = len(labels_sorted)
        probs = np.zeros((len(texts_clean), num_classes), dtype=float)

        for i, sample_scores in enumerate(outputs):
            for item in sample_scores:
                label_str = item["label"]
                score = item["score"]
                class_id = self.label2id[label_str]
                probs[i, class_id] = score

        return probs

    def predict(self, texts: List[str]) -> List[Dict]:
        """
        Возвращает список словарей по текстам:
        {
          "label_id": int,    # id класса тональности (0..num_labels-1)
          "label": str,       # строковая метка тональности, например "negative"
          "score": float      # вероятность предсказанной тональности
        }
        """
        probs = self.predict_proba(texts)
        y_pred = probs.argmax(axis=1)

        results: List[Dict] = []
        for i, class_id in enumerate(y_pred):
            score = float(probs[i, class_id])
            label_str = self.id2label[int(class_id)]

            results.append(
                {
                    "label_id": int(class_id),
                    "label": label_str,
                    "score": score,
                }
            )

        return results

    def predict_one(self, text: str) -> Dict:
        """
        Инференс тональности для одного текста.
        """
        return self.predict([text])[0]