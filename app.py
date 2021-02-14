import warnings
warnings.filterwarnings("ignore")
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import cv2
import classify
from tempfile import NamedTemporaryFile
from tensorflow.keras.preprocessing import image

st.write("""
         # COVID-19 WEBAPP
         """
        )

st.write("This is a simple image classification web app to predict wheather a person is suffering from covid-19 or not")

file = st.file_uploader("Please upload an image file", type=["jpg", "png"])
buffer = file
temp_file = NamedTemporaryFile(delete=False)
if buffer:
    temp_file.write(buffer.getvalue())
    st.write(image.load_img(temp_file.name))

if file is None:
    st.text("You haven't uploaded an image file")
else:
    image = Image.open(temp_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    st.write("")

    if st.button('predict'):
        st.write("Result...")
        label = classify.predict(temp_file)
        if label >= 0.5:
            st.write("It is Normal")
        else:
            st.write("COVID")
    

    
    