import mlflow
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import recall_score
from tqdm import tqdm

transformss = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
])

def get_data_loaders(data_dir, batch_size=32):
    train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=transformss)
    test_dataset = datasets.ImageFolder(root=f"{data_dir}/test", transform=transformss)
    val_dataset = datasets.ImageFolder(root=f"{data_dir}/val", transform=transformss)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    num_classes = len(train_dataset.classes)

    return train_loader, test_loader, val_loader, num_classes

def create_model_vgg16(num_classes, device, freeze_features=False):
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    
    if freeze_features:
        for param in model.features.parameters():
            param.requires_grad = False

    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    return model.to(device)


def create_resnet18(num_classes, device, freeze_features=False):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    if freeze_features:
        for param in model.parameters():
            param.requires_grad = False

    model.fc = nn.Sequential(nn.Linear(model.fc.in_features, num_classes),nn.Dropout(0.5))
    return model.to(device)

def train_model(model, train_loader, val_loader, criterion, optimizer,device, num_epochs=100):



    mlflow.log_param("pretrained", True)
    mlflow.log_param("optimizer", "Adam")
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("num_epochs", num_epochs)
    mlflow.log_param("batch_size", train_loader.batch_size)

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_preds = []
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

            progress_bar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Acc": f"{(correct/total):.4f}"
            })

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        epoch_recall = recall_score(all_labels, all_preds, average='macro')

        val_acc, val_recall = evaluate_model(model, val_loader,device, num_epochs, epoch)

        print(f"Epoch {epoch+1}/{num_epochs}, "
                f"Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, "
                f"Train Recall: {epoch_recall:.4f}, Val Acc: {val_acc:.4f}")

        mlflow.log_metric("train_loss", epoch_loss, step=epoch)
        mlflow.log_metric("train_accuracy", epoch_acc, step=epoch)
        mlflow.log_metric("train_recall", epoch_recall, step=epoch)
        mlflow.log_metric("val_accuracy", val_acc, step=epoch)
        mlflow.log_metric("val_recall", val_recall, step=epoch)
        #gpu_mem = torch.cuda.memory_allocated(device) / (1024 ** 2)
        #mlflow.log_metric("gpu_memory_mb", gpu_mem, step=epoch,)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            mlflow.log_artifact("best_model.pth")
    print(f"Training complete. Best Val Acc: {best_val_acc:.4f}")


    return model

def evaluate_model(model, data_loader,device, num_epochs=1, epoch=0,test=False):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        if test:
            eval_progress_bar = tqdm(data_loader, desc="Testing", leave=False)
        else:
            eval_progress_bar = tqdm(data_loader, desc=f"Validation {epoch+1}/{num_epochs}", leave=False)
        
        for inputs, labels in eval_progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

            eval_progress_bar.set_postfix({
                "Eval Acc": f"{(correct/total):.4f}"
            })
    accuracy = correct / total
    recall = recall_score(all_labels, all_preds, average='macro')  
    return accuracy, recall


def test_model(model, test_loader, device):
    test_acc, test_recall = evaluate_model(model, test_loader, device, test=True)
    print(f"Test Accuracy: {test_acc:.4f}, Test Recall: {test_recall:.4f}")
    mlflow.log_metric("test_accuracy", test_acc)
    mlflow.log_metric("test_recall", test_recall)
    print("Test metrics logged under the current run.")

