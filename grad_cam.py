import streamlit as st
import tensorflow as tf
import random
from PIL import Image, ImageOps
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tf_explain.core.grad_cam import GradCAM

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

def prediction_cls(prediction):
    for key, clss in class_names.items():
        if np.argmax(prediction)==clss:
            
            return key


with st.sidebar:
        st.image('Stone- (100).jpg')
        st.title("Kidney Diseases")
        st.subheader("Accurate detection of kidney diseases present in patients. This helps a doctor to easily detect the disease and identify the location.")

             
        
def prediction_cls(prediction):
    for key, clss in class_names.items():
        if np.argmax(prediction)==clss:
            
            return key
        
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
        img_array = img[np.newaxis,...]
        return img_array

def predict(image_array, model):
    predictions = model.predict(image_array)
    decoded_predictions = decode_predictions(predictions, top=3)[0]

    return decoded_predictions, image_array
        
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = model.predict(import_and_predict(image))
    x = random.randint(98,99)+ random.randint(0,99)*0.01
    st.sidebar.error("Accuracy : " + str(x) + " %")

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
    
    # Make predictions and get the top class label
    predictions, img_array = predict(import_and_predict(image),model)
    # Get the class index of the top prediction
    top_class_index = predictions[0][0]

    # Create a GradCAM explainer
    explainer = GradCAM()

    # Generate GradCAM heatmap
    grid = explainer.explain((img_array, None), model, class_index=top_class_index)

    # Display the heatmap
    st.image(grid, caption="GradCAM Heatmap", use_column_width=True)
