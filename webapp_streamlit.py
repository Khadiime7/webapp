import streamlit as st
from fastai.vision.all import *
from fastai.vision.all import cnn_learner, xresnet50
from lime import lime_image
import numpy as np

# Load the pretrained xresnet50 model
learn = cnn_learner(xresnet50, pretrained=True)

# Define the LimeImageExplainer
lime_explainer = lime_image.LimeImageExplainer()

def preprocess_image(image_path):
    img = PILImage.create(image_path)
    return img

def predict_fn(images):
    return learn.predict(images)[2]

def explain(image_path):
    # Preprocess the image
    img = preprocess_image(image_path)

    # Explain the image using Lime
    explanation = lime_explainer.explain_instance(img.to_tensor().permute(1, 2, 0).numpy(), predict_fn, top_labels=1, hide_color=0, num_samples=1000)

    return explanation

# Streamlit app
st.title("Explainable AI Web App")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Save the uploaded image
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.getvalue())

    # Explain the image using Lime
    lime_explanation = explain("temp_image.jpg")

    # Display the original image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Display the Lime explanation
    st.subheader("Lime Explanation:")
    lime_explanation_image = lime_explanation.image
    lime_explanation_image_clipped = np.clip(lime_explanation_image, 0.0, 1.0)
    st.image(lime_explanation_image_clipped, caption="Explanation", use_column_width=True)

    st.success("Explanation generated!")
