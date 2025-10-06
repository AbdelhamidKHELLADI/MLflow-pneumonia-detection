import cv2 as cv
import torch
from src.utils.model import create_model_vgg16
from src.utils.preprocess import contrast_stretching
import torch
import cv2 as cv
from torchvision import transforms
import click


def preprocess_image(img_path):
    """Preprocess image similar to training."""
    img = cv.imread(img_path)
    img = contrast_stretching(img)
    img = cv.resize(img, (224, 224))
    img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  
                             std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)  
    return img_tensor


def predict(model, img_path, device):
    """Predict class index for a single image."""
    model.eval()
    img_tensor = preprocess_image(img_path)
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

@click.command()

def main():

    model_path = "models/final_vgg16.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_model_vgg16(num_classes=2, device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    img_path = "/media/data/chest_xray/val/PNEUMONIA/person1946_bacteria_4875.jpeg"  # Replace with your image path
    prediction = predict(model, img_path, device)
    print("Predicted class: Normal" if prediction == 0 else "Predicted class: Pneumonia")

if __name__ == "__main__":
    main()