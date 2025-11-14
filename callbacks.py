from transformers import TrainerCallback
from clearml import Logger

class ClearMLCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return
        step = state.global_step
        acc = metrics.get("eval_accuracy")
        f1 = metrics.get("eval_f1_macro")

        # Локальный вывод метрик для отладки коллбеков
        print(f"step={step} val_acc={acc:.4f} val_f1={f1:.4f}")

        # Логирование в ClearML
        logger = Logger.current_logger()
        if logger is not None:
            if acc is not None:
                logger.report_scalar(
                    title="Validation", series="accuracy", value=acc, iteration=step
                )
            if f1 is not None:
                logger.report_scalar(
                    title="Validation", series="f1_macro", value=f1, iteration=step
                )
        else:
            print('logger недоступен и равен None')