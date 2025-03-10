import joblib
import numpy as np

# โหลดโมเดลจากไฟล์ .sav
model_1 = joblib.load("models/randomForest_model.sav")
model_2 = joblib.load("models/svm_model.sav")


# ฟังก์ชันสำหรับพยากรณ์
def predict(model_name, input_value):
    # input_value = np.array([[input_value]])  # แปลงเป็นรูปแบบที่โมเดลรับได้
    
    if model_name == "Random Forest":
        return model_1.predict(input_value)
    elif model_name == "SVM":
        return model_2.predict(input_value)
    else:
        return "❌ ไม่พบโมเดลที่เลือก"
