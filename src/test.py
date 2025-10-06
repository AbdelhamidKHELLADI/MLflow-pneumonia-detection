import mlflow.pyfunc
import pandas as pd

model = mlflow.pyfunc.load_model("models/final_resnet18_mlflow")
