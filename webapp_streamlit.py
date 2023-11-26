import streamlit as st
import tensorflow as tf
from lime import lime_image
import random
from PIL import Image, ImageOps
import numpy as np
import skimage.segmentation

import warnings
warnings.filterwarnings("ignore")


st.set_page_config(
    page_title="Explainable AI Web App",

)
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

with st.sidebar:
        st.image('Stone- (100).jpg')
        st.title("Kidney Diseases")
        st.subheader("Accurate detection of kidney diseases present in patients. This helps a doctor to easily detect the disease and identify the location.")


st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache(allow_output_mutation=True)

def load_model():
    model=tf.keras.models.load_model('kidney.h5')
    return model
    
with st.spinner('Model is being loaded..'):
    model=load_model()
    #model = keras.Sequential()
    #model.add(keras.layers.Input(shape=(224, 224, 4)))
    

st.write("""
         # Kidney diseases
         """
         )

file = st.file_uploader("", type=["jpg", "png"])

def import_and_predict(image_data):
        size = (224,224)    
        image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
        img = np.asarray(image)
        img_reshape = img[np.newaxis,...]
        return img_reshape

def lime_explain(image,model):
    # Load and preprocess the image
    img_array = import_and_predict(image)

     # Lime expects images in a different format
    img_for_lime = img_array[0].astype('double')

    # Define your prediction function
    def predict_fn(images):
        return model.predict(images)

    # Create a LimeImageExplainer
    explainer = lime_image.LimeImageExplainer()

    # Explain the prediction
    explanation = explainer.explain_instance(
        img_for_lime, 
        model.predict, 
        top_labels=5,  # Adjust as needed
        hide_color=0,
        num_features= 5,
        num_samples=1000
    )

    temp, mask = explanation.get_image_and_mask(
    explanation.top_labels[0], 
    positive_only=True, 
    num_features=5, 
    hide_rest=True)

    return explanation.image

        
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = model.predict(import_and_predict(image))
    max_value = np.max(predictions)
    max_value = round(max_value, 2)
    st.sidebar.error("Accuracy : " + str(max_value*100) + " %")

    class_names = ['Cyst','Normal','Stone','Tumor']

    string = "Detected Disease : " + class_names[np.argmax(predictions)]
    if class_names[np.argmax(predictions)] == 'Normal':
        st.balloons()
        st.sidebar.success(string)

    elif class_names[np.argmax(predictions)] == 'Cyst':
        st.sidebar.warning(string)
    
    elif class_names[np.argmax(predictions)] == 'Stone':
        st.sidebar.warning(string)
    
    elif class_names[np.argmax(predictions)] == 'Tumor':
        st.sidebar.warning(string)
    
    # Display the Lime explanation
    # lime_explanation = lime_explain(image,model)
    st.subheader("Lime Explanation:")
    st.image(lime_explain(image,model), caption="Explanation", use_column_width=True, clamp=True)
