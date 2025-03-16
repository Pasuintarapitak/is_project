import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow_datasets as tfds
from rembg import remove
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

st.title("🐶 Dog Breed Prediction App")

# Load model .keras
def load_model():
    return tf.keras.models.load_model('models/MobileNetV2_model.keras')

model = load_model()

# load datasets
def get_label_map():
    _, info = tfds.load("stanford_dogs", with_info=True)
    return info.features['label'].int2str  # convert index to breed

label_map = get_label_map()

# ฟังก์ชันปรับแต่งข้อมูล 
def preprocess_image(image):
    image = image.convert("RGB")  
    image = image.resize((128, 128))  
    image = np.array(image) / 255.0  
    image = np.expand_dims(image, axis=0)  
    return image



uploaded_file = st.file_uploader("Upload your dog picture", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
     st.image(image, caption="📸 ภาพที่อัปโหลด", use_container_width=True,width=300)
    # st.image(image, caption="📸 ภาพที่อัปโหลด", use_column_width=True,width=300)

    # remove bg
    image_no_bg = remove(image)
    # prepare for prediction
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)

    
    predicted_class = np.argmax(predictions)  
    confidence = np.max(predictions)  #ค่าความมั่นใจของ model
    predicted_label = label_map(predicted_class)

    # Result
    st.subheader("🔍 ผลการทำนาย")
    st.write(f"🐶 พันธุ์สุนัข: **{predicted_label}**")
    st.write(f"✅ ความมั่นใจ: **{confidence:.2%}**")
