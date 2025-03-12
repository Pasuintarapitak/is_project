import streamlit as st

import random
import pandas as pd

from streamlit_option_menu import option_menu
from model_handler import predict 

import numpy as np



st.title("🎯 Cancer Prediction App")
st.write("เลือกโมเดลที่ต้องการพยากรณ์ และป้อนค่าที่ต้องการ")

# Dropdown เลือกโมเดล
model_name = st.selectbox("เลือกโมเดล", ["Random Forest", "SVM"])

# รับค่าอินพุต
age = st.number_input("อายุ", min_value=20, step=1)
gender = st.selectbox("Gender ", ["Male", "Female"])
bmi = st.number_input("BMI", min_value=0.0, step=0.1)
smoke = st.radio( "Are you smoking ?", ["Yes", "No"])
genetic = st.radio(
"🧬 Genetic Risk ?",
["Indicating Low", "Indicating Medium", "Indicating High	"],
captions=[
    "No one in the family has cancer.",
    "I have a distant relative who has cancer.",
    "Someone in the family has cancer.",
],
)
# activities = st.number_input("Physical Activities (hour)", min_value=0.0, step=0.1)
activities = st.slider("🏀 Physical Activities (hour)?", 0, 10, 0 )
alcohol= st.slider("🍺 Alcohol   (consumed per week)?", 0, 5, 0 )

# alcohol = st.number_input("Alcohol   (consumed per week)", min_value=0.0, step=0.1)
history = st.radio( "💊 Cancer History ?", ["Yes", "No"])

#Convert input data to number
if gender == 'Male': gender = 0
else: gender = 1

if smoke == 'Yes': smoke = 1
else: smoke = 0

if genetic == 'Indicating Low': genetic = 0
elif genetic == 'Indicating Medium' : genetic = 1
else : genetic = 2

if history == "Yes": history = 1
else : history = 0



# inputData = { age , gender , bmi , smoke , genetic , activities , alcohol ,history }
if st.button("Predict"):
    # st.write(inputData)
    inputData = np.array([age,gender,bmi,smoke,genetic,activities,alcohol,history]).reshape(1,-1)

    if(inputData[0][0] != 0 and inputData[0][2] != 0):
        result = predict(model_name, inputData)
        if(result == 0):
            st.success(f"🩺 การวินิจฉัย : มีความเสี่ยงต่ำ ")
        else:
            st.warning(f"🩺 การวินิจฉัย : มีความเสี่ยงสูง ")  

    else:
        st.warning('กรุณากรอกข้อมูลให้ครบ', icon="⚠️")  

    