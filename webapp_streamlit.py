import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, decode_predictions, preprocess_input
from tensorflow.keras.preprocessing import image
from lime import lime_image
import numpy as np

# Load the pre-trained ResNet50 model
resnet_model = ResNet50(weights='imagenet')

# Define the explain function
def explain(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Get model predictions
    predictions = resnet_model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=3)[0]

    # Create a Lime explainer
    explainer = lime_image.LimeImageExplainer()

    # Explain the image
    explanation = explainer.explain_instance(
        img_array[0],
        resnet_model.predict,
        top_labels=1,
        hide_color=0,
        num_samples=1000
    )

    return decoded_predictions, explanation

# Streamlit app
st.title("Explainable AI Web App")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Save the uploaded image
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.getvalue())

    # Explain the image
    predictions, lime_explanation = explain("temp_image.jpg")

    # Display the original image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Display the model predictions
    st.subheader("Model Predictions:")
    for i, (imagenet_id, label, score) in enumerate(predictions):
        st.write(f"{i + 1}: {label} ({score:.2f})")

    # Display the Lime explanation
    st.subheader("Lime Explanation:")
    st.image(lime_explanation.image, caption="Explanation", use_column_width=True)

    st.success("Explanation generated!")
