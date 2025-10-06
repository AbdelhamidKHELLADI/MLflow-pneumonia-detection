# src/streamlit_app.py
import streamlit as st
import mlflow.pyfunc
import pandas as pd
import tempfile
from PIL import Image
import click
@st.cache_resource
def load_model(model_path="models/final_resnet18_mlflow"):
    model = mlflow.pyfunc.load_model(model_path)
    return model


def main():
    st.set_page_config(page_title="Chest X-Ray Classifier", layout="centered")
    st.title("🫁 Chest X-Ray Classification (Pneumonia vs Normal)")
    st.write("Upload a chest X-ray image to predict the class.")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    st.info("⚠️ Disclaimer: This tool is a research prototype. "
        "It is **not validated for medical use** and should not be considered a clinical diagnosis. "
        "For proper evaluation, please consult a healthcare professional.")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width="stretch")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image.save(tmp.name)
            tmp_path = tmp.name

        if st.button("Predict"):
            with st.spinner("Running inference..."):
                model = load_model()
                df = pd.DataFrame({"image_path": [tmp_path]})
                preds = model.predict(df)
                label = "PNEUMONIA" if preds[0] == 1 else "NORMAL"
                st.success(f"✅ Prediction: {label}")

if __name__ == "__main__":
    main()
