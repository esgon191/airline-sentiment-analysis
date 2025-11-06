# Модули
from transformers import (AutoModelForSequenceClassification, DataCollatorWithPadding,
                          TrainingArguments, Trainer, AutoTokenizer)
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from callbacks import ClearMLCallback
from clearml import Logger

# Самописные функции
from cli.parser import get_parser
from utils.utils import (make_training_arguments, get_clearml_dataset,
                   dataset_from_pandas)

logger = Logger.current_logger()

id2label = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

# Получение параметров обучения из парсера
# аргументов командной строки
parser = get_parser()
args = parser.parse_args()
train_args = make_training_arguments(args)

# Инициализация модели с huggingface
model = AutoModelForSequenceClassification.from_pretrained(
    args.model_name,
    num_labels=args.num_labels,
    id2label=id2label,
    label2id={v: k for k, v in id2label.items()},
    ignore_mismatched_sizes=True
)

# Настройки токенизации и конструктора батчей (коллатора)
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
collator = DataCollatorWithPadding(tokenizer=tokenizer)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro")
    }

# E2E получение датасета из pandas
try:
    dataset = get_clearml_dataset()
    dataset = dataset_from_pandas(
        df=dataset,
        test_size=args.test_size,
        val_size=args.val_size
        )
except NameError as e:
    raise RuntimeError(
        "dataset не определён. Создай Dataset с ключами 'train' и 'test' и объектом 'collator'. "
        "Например, через HuggingFace Datasets: train_test_split, tokenize, DataCollatorWithPadding."
    ) from e

trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["val"],
    tokenizer=tokenizer,
    data_collator=collator,
    compute_metrics=compute_metrics
)

# Добавление коллбеков для логгирования в ClearML
trainer.add_callback(ClearMLCallback())

# Обучение (Дообучение)
trainer.train()
# Замер метрик на тестовой выборке
eval_metrics = trainer.evaluate(eval_dataset=dataset["test"])


print(eval_metrics)

# Репорт метрик в clearml
for metric, value in eval_metrics.items():
    if isinstance(value, (int, float)):
        logger.report_scalar(
            title="Test",      # Группа метрик (вкладка "Scalars" -> "Test")
            series=metric,     # Название метрики
            value=value,
            iteration=0        # Поскольку series из одного шага  
        )

trainer.save_model(args.output_dir)
tokenizer.save_pretrained(args.output_dir)