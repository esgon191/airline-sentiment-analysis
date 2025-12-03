import argparse

# Значения по умолчанию
DEFAULTS = {
    # Общие
    "output_dir": "out",
    "test_size": 0.2,
    "val_size": 0.2,
    "label_column": "label",
    "text_column": "text",
    "seed": 42,

    # TF-IDF
    "tfidf_max_features": 50000,
    "tfidf_ngram_min": 1,
    "tfidf_ngram_max": 2,

    # SVC
    "svc_C": 1.0,
    "svc_kernel": "rbf",
    "svc_degree": 3,
    "svc_gamma": "scale",          # "scale" или "auto" или число
    "svc_coef0": 0.0,
    "svc_shrinking": True,
    "svc_probability": False,
    "svc_tol": 1e-3,
    "svc_cache_size": 200.0,       # MB
    "svc_class_weight": None,      # или "balanced"
}


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="training",
        description="CLI для обучения SVC-модели с TF-IDF"
    )

    # Источник данных (ClearML Dataset)
    p.add_argument(
        "--dataset_name",
        type=str,
        help="Название датасета в ClearML"
    )

    p.add_argument(
        "--dataset_version",
        type=str,
        help="Версия датасета в ClearML"
    )

    # Колонки и базовые настройки выборки
    p.add_argument(
        "--label_column",
        type=str,
        default=DEFAULTS["label_column"],
        help="Название колонки датасета с численной меткой класса"
    )

    p.add_argument(
        "--text_column",
        type=str,
        default=DEFAULTS["text_column"],
        help="Название колонки датасета с текстом"
    )

    p.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULTS["output_dir"],
        help="Путь для сохранения модели и артефактов"
    )

    p.add_argument(
        "--test_size",
        type=float,
        default=DEFAULTS["test_size"],
        help="Размер тестовой выборки относительно всей (доля)"
    )

    p.add_argument(
        "--val_size",
        type=float,
        default=DEFAULTS["val_size"],
        help="Размер валидационной выборки относительно всей (доля)"
    )

    p.add_argument(
        "--seed",
        type=int,
        default=DEFAULTS["seed"],
        help="Random seed для разбиения и обучения"
    )

    # Параметры TF-IDF
    p.add_argument(
        "--tfidf_max_features",
        type=int,
        default=DEFAULTS["tfidf_max_features"],
        help="Максимальное число признаков TF-IDF (None = без ограничения)"
    )

    p.add_argument(
        "--tfidf_ngram_min",
        type=int,
        default=DEFAULTS["tfidf_ngram_min"],
        help="Минимальный размер n-граммы для TF-IDF"
    )

    p.add_argument(
        "--tfidf_ngram_max",
        type=int,
        default=DEFAULTS["tfidf_ngram_max"],
        help="Максимальный размер n-граммы для TF-IDF"
    )

    # Параметры SVC
    p.add_argument(
        "--svc_C",
        type=float,
        default=DEFAULTS["svc_C"],
        help="Параметр регуляризации C для SVC"
    )

    p.add_argument(
        "--svc_kernel",
        type=str,
        default=DEFAULTS["svc_kernel"],
        choices=["linear", "poly", "rbf", "sigmoid"],
        help="Тип ядра SVC"
    )

    p.add_argument(
        "--svc_degree",
        type=int,
        default=DEFAULTS["svc_degree"],
        help="Степень полинома для kernel='poly'"
    )

    p.add_argument(
        "--svc_gamma",
        default=DEFAULTS["svc_gamma"],
        help="Параметр gamma для RBF/poly/sigmoid ('scale', 'auto' или число)"
    )

    p.add_argument(
        "--svc_coef0",
        type=float,
        default=DEFAULTS["svc_coef0"],
        help="Свободный член в ядре poly/sigmoid"
    )

    p.add_argument(
        "--svc_shrinking",
        type=bool,
        default=DEFAULTS["svc_shrinking"],
        help="Использовать ли shrinking-эвристику"
    )

    p.add_argument(
        "--svc_probability",
        type=bool,
        default=DEFAULTS["svc_probability"],
        help="Включить ли оценку вероятностей (увеличивает время обучения)"
    )

    p.add_argument(
        "--svc_tol",
        type=float,
        default=DEFAULTS["svc_tol"],
        help="Критерий останова обучения (tolerance)"
    )

    p.add_argument(
        "--svc_cache_size",
        type=float,
        default=DEFAULTS["svc_cache_size"],
        help="Размер кэша для ядра (MB)"
    )

    p.add_argument(
        "--svc_class_weight",
        default=DEFAULTS["svc_class_weight"],
        help="Веса классов (None или 'balanced')"
    )

    return p