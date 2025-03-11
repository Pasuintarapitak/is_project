import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow_datasets as tfds
from rembg import remove
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def load_dataset():
    dataset_name = "stanford_dogs"
    dataset, info = tfds.load(dataset_name, with_info=True, as_supervised=True)
    return dataset, info

dataset, info = load_dataset()


# แยก train และ test
train_data, test_data = dataset["train"], dataset["test"]

# image + batch size
batch_size = 32
image_size = (128, 128)  

# ปรับแต่งรูปภาพ
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

    base = tf.keras.applications.MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights="imagenet")
    base.trainable = False

    model = tf.keras.Sequential([
        base,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(120, activation="softmax")
    ])

    model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])

    class_weights = {i: 1.0 for i in range(120)} 
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    model.fit(train_data, epochs=10, validation_data=test_data, class_weight=class_weights, callbacks=[early_stopping])

    model.save('MobileNetV2_model.keras')  # savemodel
    return model

    model = train_model()