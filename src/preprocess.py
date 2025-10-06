import os
import cv2 as cv
import mlflow
from src.utils.preprocess import contrast_stretching
import click

if "MLFLOW_RUN_ID" not in os.environ:
    mlflow.set_experiment("chest-xray-pneumonia")

@click.command()
@click.option("--data_dir", type=str, default="/media/data/chest_xray", help="Path to dataset directory.")
def preprocess_folder(data_dir):
    subfolders = ["train", "test", "val"]
    classes = ["NORMAL", "PNEUMONIA"]
    print("runing")
    with mlflow.start_run(run_name="preprocessing") as run:
        
        mlflow.log_param("method", "contrast_stretching")
        for subfolder in subfolders:
            for cls in classes:
                folder_path = os.path.join(data_dir, subfolder, cls)
                for filename in os.listdir(folder_path):
                    if filename.endswith(".jpeg") or filename.endswith(".jpg") or filename.endswith(".png"):
                        img_path = os.path.join(folder_path, filename)
                        img = cv.imread(img_path)
                        processed_img = contrast_stretching(img)
                        cv.imwrite(img_path, processed_img)
        mlflow.log_artifacts(data_dir, artifact_path="preprocessed_dataset")

        print(f"✅ Preprocessing logged under run_id: {run.info.run_id}")

if __name__ == "__main__":
    preprocess_folder()