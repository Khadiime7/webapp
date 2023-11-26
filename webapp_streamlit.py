import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
# from keras.preprocessing.image import ImageDataGenerator
from lime import lime_image
import numpy as np
from PIL import Image, ImageOps

# Load your trained ResNet model
model_path = 'kidney_new.h5'  # Update with the path to your saved model
loaded_model = tf.keras.models.load_model(model_path)

# Define the LimeImageExplainer
lime_explainer = lime_image.LimeImageExplainer()

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))  # Adjust to the input size of your ResNet model
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
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

    # Make predictions using the model
    img_array = preprocess_image("temp_image.jpg")
    predictions = resnet_model.predict(img_array)
    
    # Modify the code to include an additional dimension for the batch size
    img_array_batch = np.expand_dims(img_array, axis=0)
    predictions = resnet_model.predict(img_array_batch)
    
    predicted_class = np.argmax(predictions)
    predicted_percentage = predictions[0][predicted_class]


    # Display predicted class and percentage
    st.subheader("Prediction:")
    st.write(f"Predicted Class: {predicted_class}")
    st.write(f"Prediction Percentage: {predicted_percentage * 100:.2f}%")


    st.success("Explanation generated!")
