import argparse

# Все известные параметры по умолчанию
DEFAULTS = {
    "model_name" : "tabularisai/multilingual-sentiment-analysis",
    "output_dir": "out",
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
    "test_size" : 0.2, 
    "val_size" : 0.2,
    "label_column" : "label",
    "text_column" : "text",
    "logging_strategy": "steps",
    "logging_steps": 50,
    "learning_rate": 2e-5,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 32,
    "num_train_epochs": 3,
    "weight_decay": 0.01,
    "load_best_model_at_end": True,
    "metric_for_best_model": "f1_macro",
    "num_labels": 3,
}

def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="training",
        description="CLI для обучения и оценки модели анализа тональности"
    )

    p.add_argument(
        "--model_name",
        type=str,
        default=DEFAULTS["model_name"],
        help="Модель для импорта с huggingface.co"
    )

    p.add_argument(
        "--dataset_name",
        type=str,
        help="Название датасета"
    )

    p.add_argument(
        "--dataset_version",
        type=str,
        help="Версия датасета"
    )

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
        help="Путь для сохранения модели и логов"
    )

    p.add_argument(
        "--eval_strategy",
        choices=["no", "steps", "epoch"],
        default=DEFAULTS["eval_strategy"],
        help="Частота оценки на валидации"
    )

    p.add_argument(
        "--save_strategy",
        choices=["no", "steps", "epoch"],
        default=DEFAULTS["save_strategy"],
        help="Частота сохранения модели"
    )

    p.add_argument(
        "--logging_strategy",
        choices=["no", "steps", "epoch"],
        default=DEFAULTS["logging_strategy"],
        help="Частота логирования"
    )

    p.add_argument(
        "--test_size",
        default=DEFAULTS["test_size"],
        help="Размер тестовой выборки относительно всей"
    )

    p.add_argument(
        "--val_size",
        default=DEFAULTS["val_size"],
        help="Размер валидационной выборки относительно всей"
    )

    p.add_argument(
        "--logging_steps",
        type=int,
        default=DEFAULTS["logging_steps"],
        help="Шагов между логами"
    )

    p.add_argument(
        "--learning_rate",
        type=float,
        default=DEFAULTS["learning_rate"],
        help="Скорость обучения"
    )

    p.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=DEFAULTS["per_device_train_batch_size"],
        help="Батч на устройство для обучения"
    )

    p.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=DEFAULTS["per_device_eval_batch_size"],
        help="Батч на устройство для валидации"
    )

    p.add_argument(
        "--num_train_epochs",
        type=int,
        default=DEFAULTS["num_train_epochs"],
        help="Число эпох"
    )

    p.add_argument(
        "--weight_decay",
        type=float,
        default=DEFAULTS["weight_decay"],
        help="L2 регуляризация"
    )

    p.add_argument(
        "--num_labels",
        type=int,
        default=DEFAULTS["num_labels"],
        help="Число классов целевой задачи"
    )
    
    p.add_argument(
        "--load_best_model_at_end",
        type=bool,
        default=DEFAULTS["load_best_model_at_end"],
        help="Загрузить ли для замера точности версию модели с лучшим результатом на валидации"
    )

    p.add_argument(
        "--metric_for_best_model",
        type=str,
        default=DEFAULTS["metric_for_best_model"],
        help="Метрика для определия лучшей версии модели"
    )

    p.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Включить 16-битное обучение (GPU only)"
    )

    return p
