from .cv2 import *
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as Vgg_v16_preprocess_input

st.title('Cotton Disease Exploring')

model = tf.keras.models.load_model("CNN_Cotton_Disease_project_VGG.h5")
### load file
uploaded_file = st.file_uploader("Choose a image file")


map_dict = {0:'diseased cotton leaf',
            1:'diseased cotton plant',
            2:'fresh cotton leaf',
            3:'fresh cotton plant'}

if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(opencv_image,(224,224))
    # Now do something with the image! For example, let's display it:
    st.image(opencv_image, channels="RGB")

    resized = Vgg_v16_preprocess_input(resized)
    img_reshape = resized[np.newaxis,...]

    Genrate_pred = st.button("Generate Prediction")
    if Genrate_pred:
        prediction = model.predict(img_reshape).argmax()
        st.title("Predicted Label for the image is {}".format(map_dict[prediction]))