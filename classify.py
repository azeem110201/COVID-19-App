import warnings
warnings.filterwarnings("ignore")
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import streamlit as st
from tensorflow.keras.preprocessing import image


@st.cache(allow_output_mutation=True)
def get_model():
    model = load_model('saveModel.hdf5')
    return model 

        
def predict(temp_file):
    loaded_model = get_model()
    img=image.load_img(temp_file.name, target_size=(150, 150),color_mode='rgb',grayscale=False,interpolation="nearest") 
        
    img = image.img_to_array(img)
    img = img/255
    img = np.expand_dims(img, axis=0)
    classes= loaded_model.predict(img)
    
    return classes
        
        
# =============================================================================
#         x=image.img_to_array(img)
#         x=np.expand_dims(x, axis=0)
#         images = np.vstack([x])
#      
#         classes = loaded_model.predict(images)
# =============================================================================
        
    
    
    
# =============================================================================
#     image = load_img(image, target_size=(150, 150), color_mode = "grayscale")
#     image = img_to_array(image)
#     image = image/255.0
#     image = np.reshape(image,[1,150,150,3])
#     
#     x=image.img_to_array(img)
#     x=np.expand_dims(x, axis=0)
#     images = np.vstack([x])
#   
#     classes = model.predict(images, batch_size=10)
# 
#     classes = loaded_model.predict_classes(image)
# =============================================================================
