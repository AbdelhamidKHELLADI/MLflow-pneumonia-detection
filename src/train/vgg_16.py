import mlflow
from mlflow.models.signature import infer_signature
import torch
import torch.nn as nn
import torch.optim as optim
import click
import os
from src.utils.model import get_data_loaders, create_model_vgg16, train_model, test_model
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
    model = create_model_vgg16(num_classes, device,freeze_features=freeze)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    name="vgg16_finetune_full" if freeze else "vgg16_finetune" 
    with mlflow.start_run(run_name=name,nested=True):
        mlflow.log_param("finetune_only_last", freeze)
        trained_model = train_model(model, train_loader, test_loader, criterion, optimizer,device, num_epochs=epochs)
        test_model(trained_model, val_loader, device)
        if freeze:
            model_path="models/final_vgg16_freeze.pth"
        else:
            model_path="models/final_vgg16.pth"
        torch.save(trained_model, model_path)
        print("Training complete and model saved.")

        mlflow.log_artifact(model_path)
        
        example_inputs, _ = next(iter(val_loader))
        example_inputs = example_inputs.to(device)
        example_outputs = trained_model(example_inputs)

        signature = infer_signature(
            example_inputs.cpu().numpy(),
            example_outputs.cpu().detach().numpy()
        )

        mlflow.pytorch.log_model(
            pytorch_model=trained_model,
            artifact_path="model",
            registered_model_name="ChestXrayVGG16",
            signature=signature,
            input_example=example_inputs[0].cpu().numpy()
        )

        mlflow.pytorch.save_model(trained_model, model_path.split('.')[0])



if __name__ == "__main__":
    main()
    