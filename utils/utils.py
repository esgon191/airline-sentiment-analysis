from clearml import Dataset
import pandas as pd
import os, argparse, transformers, datasets


def get_clearml_dataset(
        dataset_name : str,
        dataset_version : str,
        dataset_project : str = 'airline-sentiment-analysis'
) -> pd.DataFrame:
    ds = Dataset.get(
        dataset_name=dataset_name,
        dataset_project=dataset_project,
        dataset_version=dataset_version
    )

    # Путь к папке с датасетом
    path = ds.get_local_copy()

    # Пока что весь датасет — один csv
    # Поведение будет модифицировано при переезде на S3 и паркеты
    df = pd.read_csv(os.path.join(path, os.listdir(path)[0]))

    return df


def make_training_arguments(
        args: argparse.Namespace
) -> transformers.TrainingArguments:
    """
    Делает из полученных параметров запуска
    параметры обучения, которые хавает transformers
    """
    # Определение направления роста метрики
    metric = getattr(args, "metric_for_best_model", None)
    gib = getattr(args, "greater_is_better", None)
    if gib is None and isinstance(metric, str) and metric:
        gib = not metric.endswith("loss")

    kwargs = dict(
        output_dir=args.output_dir,
        eval_strategy=args.eval_strategy,  
        save_strategy=args.save_strategy,
        logging_strategy=args.logging_strategy,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
    )

    if gib is not None:
        kwargs["greater_is_better"] = gib

    train_args = transformers.TrainingArguments(**kwargs)

    return train_args


def dataset_from_pandas(
        df: pd.DataFrame,
        test_size: float,
        val_size: float,
        text_column: str = "text",
        label_column: str = "label",
        seed: int = 42
) -> datasets.DatasetDict:
    """
    Возвращает ожидаемый библиотекой transformers dataset
    разбитый на train, val, test части
    """
    df[label_column] = df[label_column].astype(int)

    dataset = datasets.Dataset.from_pandas(df[[text_column, label_column]], preserve_index=False)
 
    # Первичное train-test разделение
    split_1 = dataset.train_test_split(test_size=test_size, seed=seed)
    train_val_ds = split_1["train"]
    test_ds = split_1["test"]

    # Отделение валидационной части
    val_rel = val_size / (1.0 - test_size)
    split_2 = train_val_ds.train_test_split(test_size=val_rel, seed=seed)
    train_ds = split_2["train"]
    val_ds = split_2["test"]

    return datasets.DatasetDict({
        "train": train_ds,
        "val": val_ds,
        "test": test_ds,
    })