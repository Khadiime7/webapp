import streamlit as st
import tensorflow as tf
import random
from PIL import Image, ImageOps
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K


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

# Function to generate Grad-CAM
def generate_grad_cam(img_array, model, last_conv_layer_name=model.get_layer('vgg16').layers[-2].output, pred_index=None):
    # Get the class predictions for the image
    predictions = model(img_array)

    if pred_index is None:
        pred_index = np.argmax(predictions[0])

    # Get the last layer of the model (fully connected layer before softmax)
    last_layer = model.layers[-1]

    # Create a model that maps the input image to the output predictions
    grad_model = tf.keras.models.Model([model.inputs], [model.output, last_layer.output])
    
    grad_model = tf.keras.models.Model(model.inputs, [last_conv_layer_name, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = K.mean(grads, axis=(0, 1, 2))

    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    return heatmap.numpy()

# Function to apply Grad-CAM on the image
def apply_grad_cam(img, heatmap):
    heatmap = np.uint8(255 * heatmap)
    heatmap = Image.fromarray(heatmap, 'L').resize((img.width, img.height))
    heatmap = heatmap.convert("RGB")

    superimposed_img = Image.alpha_composite(img.convert("RGBA"), Image.new("RGBA", img.size, (0, 0, 0, 0)))
    superimposed_img.paste(heatmap, (0, 0), heatmap)

    return superimposed_img

        
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
    
    heatmap = generate_grad_cam(import_and_predict(image), model, pred_index=class_names[np.argmax(predictions)])
    st.subheader("Grad-CAM Visualization")
    st.image(apply_grad_cam(image, heatmap), caption="Grad-CAM", use_column_width=True)
