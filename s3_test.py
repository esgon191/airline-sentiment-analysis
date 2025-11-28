from clearml import Task, Logger, Dataset, StorageManager
from pathlib import Path
from datetime import datetime

task = Task.init(
    project_name = "airline review sentiment analysis",
    task_name="ClearML S3 connection test",
    output_uri="s3://s3.ru-7.storage.selcloud.ru:443/kaftaranov-mlops-kursovaya/",
    task_type=Task.TaskTypes.testing
)

try:
    Logger.current_logger().report_text(f"Test task initialized successfully {datetime.now()}")
    print("Успешно подключено к ClearML")

    task.upload_artifact(
        name="s3_test",
        artifact_object="README.md"
    )
    print('Артефакт загружен в S3')

except Exception as e:
    print(e)

finally:
    task.close()
    print("Успешное завершение задачи")