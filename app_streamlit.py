import streamlit as st
from captum.attr import IntegratedGradients
import torch
import torchvision.transforms as transforms
from PIL import Image

# Load the pretrained ResNet model
resnet_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
resnet_model.eval()

# Create an instance of the IntegratedGradients algorithm
integrated_gradients = IntegratedGradients(resnet_model)

# Define the transformation
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Define the explain function
def explain(image_tensor):
    # Get model predictions
    output = resnet_model(image_tensor)

    # Compute attributions using Integrated Gradients
    attributions = integrated_gradients.attribute(image_tensor)

    return attributions

# Streamlit app
st.title("Explainable AI Web App")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Preprocess the uploaded image
    image = Image.open(uploaded_file)
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    # Explain the image
    attributions = explain(input_batch)

    # Display the original image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # You can use attributions for further analysis or visualization

    st.success("Explanation generated!")
