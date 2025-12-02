from clearml import Dataset
import pandas as pd
import os, argparse
from sklearn.model_selection import train_test_split


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


def dataset_from_pandas(
        df: pd.DataFrame,
        test_size: float,
        val_size: float,
        text_column: str,
        label_column: str,
        seed: int = 42
) -> tuple:
    """
    Возвращает 6 pd.Series: X, y - train, val, test
    """
    # Отбор только нужных для обучения колонок
    df = df[[text_column, label_column]]

    df = df[[text_column, label_column]].copy()
    df[label_column] = df[label_column].astype(int)

    # 1 Отделение test-части
    df_train_val, df_test = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df[label_column],
    )

    # 2 — доля валидации внутри train_val
    val_rel = val_size / (1.0 - test_size)

    # 3 Отделение val-части
    df_train, df_val = train_test_split(
        df_train_val,
        test_size=val_rel,
        random_state=seed,
        stratify=df_train_val[label_column],
    )

    X_train = df_train[text_column].values
    y_train = df_train[label_column].values

    X_val = df_val[text_column].values
    y_val = df_val[label_column].values

    X_test = df_test[text_column].values
    y_test = df_test[label_column].values

    return X_train, X_val, X_test, y_train, y_val, y_test