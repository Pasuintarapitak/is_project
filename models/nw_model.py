import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow_datasets as tfds
from rembg import remove
import os
# from tensorflow. import mixed_precision

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# import os
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# import tensorflow as tf

def load_dataset():
    dataset_name = "stanford_dogs"
    dataset, info = tfds.load(dataset_name, with_info=True, as_supervised=True)
    return dataset, info

dataset, info = load_dataset()
# print(info)

# แยก train และ test
train_data, test_data = dataset["train"], dataset["test"]

# กำหนดขนาดภาพและbatch
batch_size = 32
image_size = (128, 128)  

# ฟังก์ชันปรับแต่งข้อมูล (Preprocessing)
def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0  # Normalize
    image = tf.image.resize(image, image_size)  # Resize
    return image, label

# ชุดข้อมูลสำหรับการTrain
train_data = (
    train_data
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)  # Preprocess ก่อน
    .shuffle(1000)
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)

# # ฟังก์ชันเพิ่มข้อมูล(Augmentation)
# def augment(image, label):
#     image = tf.image.random_flip_left_right(image)
#     image = tf.image.random_brightness(image, max_delta=0.1)
#     image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
#     image = tf.image.random_hue(image, max_delta=0.02)
#     return image, label 

# ชุดข้อมูลสำหรับการTest
test_data = (
    test_data
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)


# โหลดโมเดลที่ฝึกเสร็จแล้ว
def load_model():
    model = tf.keras.models.load_model('MobileNetV2_model.keras')
    return model

def train_model():
    # ใช้ EfficientNetB0 แทน MobileNetV2
    # base_model = tf.keras.applications.EfficientNetB0(input_shape=(128, 128, 3), include_top=False, weights="imagenet")

    # mixed_precision.set_global_policy('mixed_float16')

    base_model = tf.keras.applications.MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights="imagenet")

    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(120, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

  
    # แก้ไข class_weights ให้ตรงกับจำนวนคลาสในชุดข้อมูล (120 คลาส)
    class_weights = {i: 1.0 for i in range(120)}  # หรือปรับตามการกระจายคลาสที่ต้องการ
    # ปรับ class_weight ตามที่ต้องการ
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    model.fit(train_data, epochs=10, validation_data=test_data, class_weight=class_weights, callbacks=[early_stopping])


    model.save('MobileNetV2_model.keras')  # บันทึกโมเดลที่ฝึกเสร็จแล้ว
    return model

    model = train_model()