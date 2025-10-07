# 🫁 Chest X-Ray Pneumonia Classifier  

## 📘 Description

This project implements a deep learning pipeline for classifying chest X-ray images into Normal or Pneumonia categories.
It is built using PyTorch, MLflow, and Streamlit.

⚠️ **Disclaimer**:  
This project is for **research and educational purposes only**.  
It is **not validated for medical diagnosis** and must not be used as a substitute for professional healthcare advice.  

## Key features:

**Preprocessing pipeline** (contrast stretching for image enhancement)

**Model training using VGG16 and ResNet18**

**MLflow tracking for experiments, models, and artifacts**

**Streamlit web app for easy model deployment and visualization**

## ⚙️ Project Structure
```bash
MLflow-pneumonia-detection/
├── MLproject                 # MLflow project configuration
├── conda_env.yml             # Conda environment for MLflow runs
├── requirements.txt          
├── models/                   # Saved and logged models
├── src/
│   ├── train/
│   │   ├── vgg_16.py
│   │   └── resnet_18.py
│   ├── preprocess.py
│   ├── utils/
│   └── streamlit_app.py
├── mlruns/                   # MLflow experiment tracking (will be created after running the training scripts)
└── README.md
```
## 📂 Dataset

The project uses the Chest X-Ray Images (Pneumonia) dataset from Kaggle.

 Dataset name: Chest X-Ray Images (Pneumonia)
 Source: [Kaggle Dataset Link](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

### Download Instructions

* Visit the dataset page on Kaggle (link above).
* Click “Download” and unzip the file.
* Ensure the structure looks like this:
```bash
  chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

## 🚀 Getting Started
before 
### 1. Clone the Repository
```bash
git clone https://github.com/AbdelhamidKHELLADI/MLflow-pneumonia-detection.git
cd MLflow-pneumonia-detection
```
 ### 2. Data Preprocessing

Before training, enhance your dataset using contrast stretching.
Run with MLflow:
```bash
mlflow run . -e preprocess -P data_dir=/path/to/chest_xray
```
This will:
* Apply image enhancement (contrast stretching)
* Log preprocessing metadata and dataset snapshot to MLflow

## 4. Train Models

You can fine-tune either VGG16 or ResNet18 using MLflow entry points.

### Train VGG16
```bash
mlflow run . -e train_vgg \
  -P data_dir=/path/to/chest_xray \
  -P epochs=25 \
  -P freeze=False
```
### Train ResNet18
```bash
mlflow run . -e train_resnet \
  -P data_dir=/path/to/chest_xray \
  -P epochs=25 \
  -P freeze=True
```
* `--freeze` explication:  

| Option  | Description                                                                  |
| :------ | :--------------------------------------------------------------------------- |
| `True`  | Freeze pretrained layers (only train classifier). Faster, less overfitting.  |
| `False` | Train all layers (full fine-tuning). Slower, more accurate with enough data. |

* These commands:
  * Train and evaluate the model
  * Log metrics, parameters, and artifacts to MLflow
  * Save models to the models/ directory

## 💻 5. Launch the Streamlit App

Once a model is trained and saved, you can run the Streamlit web interface for predictions:

```bash
export PYTHONPATH=$(pwd)
export MODEL_PATH=/exp/path/to/models/resnet18  # optional: set custom model for inference
streamlit run src/streamlit_app.py
```

Then open your browser at:
👉 http://localhost:8501

Upload a chest X-ray image (.jpg, .jpeg, .png) and get predictions for:

NORMAL or PNEUMONIA

## Example Workflow

* Preprocess data
```bash
mlflow run . -e preprocess -P data_dir=/data/chest_xray
```
* Train ResNet18
```bash
mlflow run . -e train_resnet -P data_dir=/data/chest_xray -P freeze=False
```
* Launch the web app
```bash
streamlit run src/streamlit_app.py
```

## Tracking Experiments
* MLflow automatically tracks:
* Parameters (learning rate, freeze state, epochs, etc.)
* Metrics (accuracy, recall, loss)
* Models and artifacts

To view the MLflow UI locally:
```bash
mlflow ui
```

Then open: http://localhost:5000


## License

This project is licensed under the MIT License — feel free to use and modify it for your research or educational work.
