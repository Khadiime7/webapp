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
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Get model predictions
    predictions = resnet_model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=3)[0]

    # Create a Lime explainer
    lime_explainer = lime_image.LimeImageExplainer()
    lime_explanation = lime_explainer.explain_instance(
        img_array[0],
        resnet_model.predict,
        top_labels=1,
        hide_color=0,
        num_samples=1000
    )

    # Clip pixel values to [0.0, 1.0]
    img_array_clipped = np.clip(img_array[0], 0.0, 1.0)

    return decoded_predictions, lime_explanation, img_array_clipped

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
