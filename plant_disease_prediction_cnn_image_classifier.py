import streamlit as st
import os
import json
from zipfile import ZipFile
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import kagglehub

# Ensure necessary directories exist
os.makedirs("datasets", exist_ok=True)
os.makedirs("model", exist_ok=True)

# Set global variables
DATASET_PATH = "datasets/plantvillage-dataset.zip"
EXTRACT_PATH = "datasets/plantvillage-dataset"
MODEL_PATH = "model/plant_disease_prediction_model.h5"
CLASS_INDICES_PATH = "model/class_indices.json"

# Streamlit Interface
st.title("Plant Disease Prediction - End to End")
st.write("This app predicts plant diseases using a pre-trained Convolutional Neural Network.")

# Dataset Download and Extraction with kagglehub
if st.sidebar.button("Download and Extract Dataset"):
    st.sidebar.write("Downloading and extracting dataset using kagglehub...")
    dataset_path = kagglehub.dataset_download("abdallahalidev/plantvillage-dataset", dest_dir="datasets")
    if dataset_path:
        with ZipFile(dataset_path, "r") as zip_ref:
            zip_ref.extractall(EXTRACT_PATH)
        st.sidebar.success("Dataset downloaded and extracted!")
    else:
        st.sidebar.error("Failed to download dataset. Ensure Kaggle credentials are configured.")

# Train Model
if st.sidebar.button("Train Model"):
    if not os.path.exists(EXTRACT_PATH):
        st.sidebar.error("Dataset not found. Please download and extract it first.")
    else:
        st.sidebar.write("Training model...")

        # Data Preprocessing
        base_dir = os.path.join(EXTRACT_PATH, "color")
        img_size = 224
        batch_size = 32

        data_gen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

        train_generator = data_gen.flow_from_directory(
            base_dir,
            target_size=(img_size, img_size),
            batch_size=batch_size,
            subset="training",
            class_mode="categorical"
        )

        validation_generator = data_gen.flow_from_directory(
            base_dir,
            target_size=(img_size, img_size),
            batch_size=batch_size,
            subset="validation",
            class_mode="categorical"
        )

        # Model Definition
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=(img_size, img_size, 3)),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D(2, 2),
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.Dense(train_generator.num_classes, activation="softmax")
        ])

        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

        model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // batch_size,
            epochs=5,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // batch_size
        )

        # Save Model and Class Indices
        model.save(MODEL_PATH)
        with open(CLASS_INDICES_PATH, "w") as f:
            json.dump({v: k for k, v in train_generator.class_indices.items()}, f)

        st.sidebar.success("Model trained and saved!")

# Load Model for Predictions
@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASS_INDICES_PATH):
        return None, None
    model = load_model(MODEL_PATH)
    with open(CLASS_INDICES_PATH, "r") as f:
        class_indices = json.load(f)
    return model, {v: k for k, v in class_indices.items()}

# Prediction Section
st.header("Predict Plant Disease")
uploaded_image = st.file_uploader("Upload an image for prediction", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    model, class_indices = load_trained_model()

    if model is None:
        st.error("Model not found. Train the model first.")
    else:
        # Preprocess the image
        def preprocess_image(img, target_size=(224, 224)):
            img = img.resize(target_size)
            img_array = np.array(img).astype("float32") / 255.0
            return np.expand_dims(img_array, axis=0)

        preprocessed_img = preprocess_image(image)
        predictions = model.predict(preprocessed_img)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class_name = class_indices[predicted_class_index]

        st.success(f"Predicted Class: **{predicted_class_name}**")
