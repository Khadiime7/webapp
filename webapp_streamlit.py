import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from lime import lime_image
import numpy as np

# Load your trained ResNet model
model_path = 'path/to/your/saved_model'  # Update with the path to your saved model
loaded_model = tf.keras.models.load_model(model_path)

# Define the LimeImageExplainer
lime_explainer = lime_image.LimeImageExplainer()

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def predict_fn(images):
    return loaded_model(images)

def explain(image_path):
    # Preprocess the image
    img_array = preprocess_image(image_path)

    # Explain the image using Lime
    explanation = lime_explainer.explain_instance(img_array[0], predict_fn, top_labels=1, hide_color=0, num_samples=1000)

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
