import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import tensorflow_datasets as tfds

# โหลดโมเดลที่ฝึกเสร็จแล้ว
def load_model():
    return tf.keras.models.load_model('models/MobileNetV2_model.keras')

model = load_model()

# โหลดข้อมูลชุด `stanford_dogs` เพื่อดึงชื่อพันธุ์สุนัข
def get_label_map():
    _, info = tfds.load("stanford_dogs", with_info=True)
    return info.features['label'].int2str  # ฟังก์ชันแปลง index → ชื่อพันธุ์

label_map = get_label_map()

def preprocess_image(image):
    image = image.convert("RGB")  # แปลงเป็น RGB
    image = image.resize((128, 128))  # ปรับขนาดให้ตรงกับโมเดล
    image = np.array(image) / 255.0  # Normalize เป็น 0-1
    image = np.expand_dims(image, axis=0)  # เพิ่มมิติให้เป็น (1, 128, 128, 3)
    return image

# UI สำหรับ Streamlit
st.title("🐶 ทำนายพันธุ์สุนัขด้วย AI")

uploaded_file = st.file_uploader("อัปโหลดภาพสุนัข", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="📸 ภาพที่อัปโหลด", use_column_width=True)

    # เตรียมข้อมูลและทำนาย
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    
    # ดึงค่าคลาสที่โมเดลพยากรณ์ได้
    predicted_class = np.argmax(predictions)  # ค่าคลาสที่มีค่าความน่าจะเป็นสูงสุด
    confidence = np.max(predictions)  # ค่าความมั่นใจ
    predicted_label = label_map(predicted_class)

    # แสดงผลลัพธ์
    st.subheader("🔍 ผลการทำนาย")
    st.write(f"🐶 พันธุ์สุนัข: **{predicted_label}**")
    st.write(f"✅ ความมั่นใจ: **{confidence:.2%}**")
