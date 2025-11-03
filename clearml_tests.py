from clearml import Task, Logger
from datetime import datetime

try:
    task = Task.init(
        project_name = "airline review sentiment analysis",
        task_name="Тестирование подключение к ClearML из git-repo",
        task_type=Task.TaskTypes.testing,
        output_uri=False
    )
    Logger.current_logger().report_text(f"Test task initialized successfully {datetime.now()}")
    print("Успешно подключено к ClearML")

except Exception as e:
    print(e)

finally:
    task.close()
    print("Успешное завершение задачи")