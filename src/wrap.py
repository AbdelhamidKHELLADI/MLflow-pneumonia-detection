import mlflow.pyfunc
from src.utils.mlflow_wrap import ChestXrayModel


mlflow.pyfunc.save_model(
    path="models/final_resnet18_freeze_mlflow_pre",
    python_model=ChestXrayModel(),
    artifacts={"pytorch_model": "models/final_resnet18_freeze_mlflow"}
)
