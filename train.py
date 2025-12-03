# Модули
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC
from clearml import Logger, Task
import os, joblib

# Самописные функции
from cli.parser import get_parser
from utils.utils import (get_clearml_dataset,
                   dataset_from_pandas, tf_idf_vectorise)


# Инициализация ClearML задачи
task = Task.init(
    project_name="airline-sentiment-analysis",
    task_name="train-multilingual-sentiment",
    task_type=Task.TaskTypes.training,  # можно опустить, но так явнее
)

logger = Logger.current_logger()

# Получение параметров обучения из парсера
# аргументов командной строки
parser = get_parser()
args = parser.parse_args()

# Логирование конфига в ClearML
task.connect(args)

# Инициализация модели 
model = SVC(
    C=args.svc_C,
    kernel=args.svc_kernel,
    degree=args.svc_degree,
    gamma=args.svc_gamma,
    coef0=args.svc_coef0,
    shrinking=args.svc_shrinking,
    probability=args.svc_probability,
    tol=args.svc_tol,
    cache_size=args.svc_cache_size,
    class_weight=args.svc_class_weight
)

# E2E получение датасета из pandas
try:
    dataset = get_clearml_dataset(
        dataset_name=args.dataset_name,
        dataset_version=args.dataset_version 
    )
    X_train, X_val, X_test, y_train, y_val, y_test = dataset_from_pandas(
        df=dataset,
        test_size=args.test_size,
        val_size=args.val_size,
        text_column=args.text_column,
        label_column=args.label_column
    )
    X_train, X_val, X_test, y_train, y_val, y_test = tf_idf_vectorise(X_train, X_val, X_test, y_train, y_val, y_test)

except NameError as e:
    raise RuntimeError(
        "dataset не определён. Создай Dataset с ключами 'train' и 'test' и объектом 'collator'. "
        "Например, через HuggingFace Datasets: train_test_split, tokenize, DataCollatorWithPadding."
    ) from e

model.fit(X_train, y_train)

y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_f1_macro = f1_score(y_test, y_test_pred, average="macro")

logger.report_scalar("Metrics", "test_accuracy", test_accuracy, iteration=0)
logger.report_scalar("Metrics", "test_f1_macro", test_f1_macro, iteration=0)

joblib.dump(model, "svc_model.joblib")

task.upload_artifact(
    name="svc_model_file",
    artifact_object=None,
    filename="svc_model.joblib"
)

task.close()