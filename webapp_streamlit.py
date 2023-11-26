import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
# from keras.preprocessing.image import ImageDataGenerator
from lime import lime_image
import numpy as np
from PIL import Image, ImageOps

# Load your trained ResNet model
model_path = 'kidney.h5'  # Update with the path to your saved model
resnet_model = tf.keras.models.load_model(model_path)

# Define the explain function
def explain(image_path):
    # Load and preprocess the image
    img_array = preprocess_image(image_path)

    # Get predictions from the model
    predictions = resnet_model.predict(img_array)

    # Get the top predicted class and its probability
    top_class = np.argmax(predictions)
    top_probability = predictions[0, top_class]

    # Print or use the top class and probability as needed
    print(f"Top Class: {top_class}, Probability: {top_probability}")

    # Explain the image using Lime
    explanation = lime_explainer.explain_instance(img_array[0], predict_fn, top_labels=1, hide_color=0, num_samples=1000)

    return predictions, explanation, img_array

# Streamlit app
st.title("Explainable AI Web App")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Save the uploaded image
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.getvalue())

    # Explain the image
    predictions, lime_explanation, img_array_clipped = explain("temp_image.jpg")

    # Display the original image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Display the model predictions
    st.subheader("Model Predictions:")
    for i, (imagenet_id, label, score) in enumerate(predictions):
        st.write(f"{i + 1}: {label} ({score:.2f})")

    # Display the Lime explanation
    st.subheader("Lime Explanation:")
    st.image(lime_explanation.image, caption="Explanation", use_column_width=True, clamp=True)

 
    st.success("Explanation generated!")
