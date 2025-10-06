
import mlflow.pyfunc
import mlflow.pytorch
import torch
import cv2 as cv
from PIL import Image
from torchvision import transforms
from src.utils.preprocess import contrast_stretching

class ChestXrayModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # mlflow.pytorch.load_model(context.artifacts["pytorch_model"]) works if artifact is an mlflow model dir
        self.pytorch_model = mlflow.pytorch.load_model(context.artifacts["pytorch_model"])
        self.pytorch_model.to(self.device)
        self.pytorch_model.eval()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def _preprocess_path(self, path):
        img = cv.imread(path)
        if img is None:
            raise ValueError(f"Cannot read image: {path}")
        img = contrast_stretching(img)
        img = cv.resize(img, (224, 224))
        # ensure 3 channels
        if img.ndim == 2:
            img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        else:
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        pil = Image.fromarray(img)
        t = self.transform(pil).unsqueeze(0).to(self.device)
        return t

    def predict(self, context, model_input):
        # model_input is expected to be a pandas DataFrame with column "image_path"
        if isinstance(model_input, str):
            paths = [model_input]
        else:
            # if DataFrame/Series passed, extract column if exists
            try:
                paths = list(model_input["image_path"])
            except Exception:
                # fallback: iterable of strings
                paths = list(model_input)

        tensors = [self._preprocess_path(p) for p in paths]
        batch = torch.cat(tensors, dim=0)
        with torch.no_grad():
            outputs = self.pytorch_model(batch)
            preds = torch.argmax(outputs, dim=1).cpu().numpy().tolist()
        return preds

if __name__ == "__main__":
    print("it works")
