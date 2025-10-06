import mlflow
import mlflow.pyfunc
from mlflow.models.signature import infer_signature
from mlflow.models.signature import ModelSignature
from mlflow.types import Schema, ColSpec
import torch
import torch.nn as nn
import torch.optim as optim
from src.utils.mlflow_wrap import ChestXrayModel
from src.utils.model import get_data_loaders, create_resnet18, train_model, test_model
import click
import os

if "MLFLOW_RUN_ID" not in os.environ:
    mlflow.set_experiment("chest-xray-pneumonia")
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@click.command()
@click.option("--data_dir", type=str, default="/media/data/chest_xray", help="Path to dataset directory.")
@click.option("--epochs", type=int, default=25, help="Number of training epochs.")
@click.option("--freeze", type=str, default="false", help="Freeze all layers except classifier.")

def main(data_dir, epochs, freeze):
    freeze=freeze.lower() in ["1","t", "true", "y", "yes"]
    data_dir = data_dir
    train_loader, test_loader, val_loader, num_classes = get_data_loaders(data_dir, batch_size=32)
    model = create_resnet18(num_classes, device,freeze_features=freeze)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    with mlflow.start_run(run_name="ResNet_finetune_full"):
        mlflow.log_param("model", "ResNet18")
        mlflow.log_param("finetune_only_last", freeze)

        trained_model = train_model(model, train_loader, test_loader, criterion, optimizer, device, num_epochs=epochs)
        test_model(trained_model, val_loader, device)

        torch.save(trained_model.state_dict(), "models/final_resnet18.pth")
        mlflow.log_artifact("models/final_resnet18.pth")
        print("Training complete and model saved.")

        example_inputs, _ = next(iter(val_loader))
        example_inputs = example_inputs.to(device)
        example_outputs = trained_model(example_inputs)

        signature = infer_signature(
            example_inputs.cpu().numpy(),
            example_outputs.cpu().detach().numpy()
        )

        mlflow.pytorch.log_model(
            pytorch_model=trained_model,
            name="model",
            registered_model_name="ChestXrayResNet",
            signature=signature,
            input_example=example_inputs[0].cpu().numpy()
        )


        mlflow.pytorch.save_model(trained_model, "models/final_resnet18_mlflow_base")

        
        input_schema = Schema([ColSpec("string", "image_path")])
        output_schema = Schema([ColSpec("long", "prediction")])
        pyfunc_signature = ModelSignature(inputs=input_schema, outputs=output_schema)

        mlflow.pyfunc.save_model(
            path="models/final_resnet18_mlflow",
            python_model=ChestXrayModel(),
            artifacts={"pytorch_model": "models/final_resnet18_mlflow_base"},  
            signature=pyfunc_signature
        )


        print("Model saved with preprocessing and logged to MLflow.")


if __name__ == "__main__":
    main()