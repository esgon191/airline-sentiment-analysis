def make_preprocess_fn(tokenizer, label2id=None):
    """
    Фабрика функций-токенизаторов для transformers-моделей
    """
    def preprocess(batch: dict) -> dict:
        """
        Токенизирует батч данных перед передачей в transformers-модель
        """
        enc = tokenizer(batch["text"], truncation=True, padding=False)
        if label2id is not None:
            enc["labels"] = [label2id[v] for v in batch["label"]]
        else:
            enc["labels"] = batch["label"]
        return enc
    return preprocess