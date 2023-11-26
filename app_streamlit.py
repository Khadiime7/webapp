import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from lime import lime_image

# Load the pretrained ResNet model
resnet_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
resnet_model.eval()

# Define the transformation
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Define the explain function using Lime
def explain(image):
    # Preprocess the image
    input_tensor = preprocess(image).unsqueeze(0)

    # Get model predictions
    output = resnet_model(input_tensor)

    # Create a Lime explainer
    explainer = lime_image.LimeImageExplainer()

    # Explain the image
    explanation = explainer.explain_instance(input_tensor[0].permute(1, 2, 0).numpy(), resnet_model.predict, top_labels=1, hide_color=0, num_samples=1000)

    return explanation

# Streamlit app
st.title("Explainable AI Web App")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Preprocess the uploaded image
    image = Image.open(uploaded_file)

    # Explain the image
    explanation = explain(image)

    # Display the original image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Display the explanation
    st.image(explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5)[0], caption="Explanation", use_column_width=True)

    st.success("Explanation generated!")
