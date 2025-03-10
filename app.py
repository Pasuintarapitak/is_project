from pyexpat import model

from sklearn.metrics import classification_report, confusion_matrix
import streamlit as st

import random
import pandas as pd

from streamlit_option_menu import option_menu
from model_handler import predict  # ดึงฟังก์ชันพยากรณ์


st.set_page_config(page_title="My App", page_icon="🌟", layout="wide")
nsafe_allow_html=True

# สร้าง Navbar แบบแนวนอน
selected = option_menu(
    menu_title=None,  
    options=["Machine Learning Documentation", "Neural Network Documentation"],  
    icons=["info-circle", "info-circle"],  
    menu_icon="cast",  
    default_index=0,  
    orientation="horizontal",
)


if selected == "Machine Learning Documentation":
    # st.title()
   
    #Header
    st.title(":orange[Cancer] Prediction Model  :sunglasses:")
    st.write("โมเดลทำนายโอกาสการเป็นมะเร็ง")

    #Dataset
    df = pd.read_csv("Dataset/cancer_dataset.csv")
    st.dataframe(df)
    st.markdown("***การวินิจฉัยโอกาสการเป็นมะเร็งของผู้ป่วยจาก ข้อมูลผู้ป่วย เช่น อายุ เพศ bmi และอื่นๆ รวมทั้วการทำกิจกรรมต่างๆ ซึ่งมีส่งผลต่อความเสี่ยง***")
    st.markdown("***Raw Dateset : https://www.kaggle.com/datasets/rabieelkharoua/cancer-prediction-dataset/data***")
   
    #Feature
    st.subheader("Features Explanation:", divider="red")
    feature = pd.DataFrame(
    {
        "Features": ["Age", "Gender","BMI","GeneticRisk","PhysicalActivity",
                         "AlcoholIntake","CancerHistory","Diagnosis"],
        "Meaning": ["อายุ   ( ช่วง 20 - 80 ปี )", 
                    "เพศ :  Male(ชาย)  ,  Female(หญิง)",
                    "ค่าดัชนีมวลกาย    (กิโลกรัม)",
                    "ความเสี่ยงทางพันธุกรรม ระดับ: (  เสี่ยงต่ำ ,เสี่ยงกลาง ,เสี่ยงสูง  )",
                    "การออกกำลัง  (ชั่วโมง)",
                    "การดื่มแอลกอฮอล์  (ต่อสัปดาห์)",
                    "ประวัติการรักษาโรคมะเร็ง",
                    "การวินิจฉัย (Outputs)"],

    }
    )
    # st.dataframe(feature, height=300, width=600, use_container_width=True)
    feature = feature.reset_index(drop=True) 
    st.dataframe(feature,height=320, width=1200)


    #prepare data
    st.title(":red[Prepare] dataset")

    st.write("1. ทำการดึง dataset จาก ไฟล์ .csv")
    code = '''
        data = pd. read_csv("/content/cancer_dataset.csv") 
    '''
    st.code(code, language="python")


    st.write("2. ทำการลบแถวที่มี NULL และแปลงข้อมูลเป็นตัวเลขสำหรับการคำนวณ")
    code = '''
        #Drop data which row has null
        data = data.dropna()
        
        #Convert data type to number for evaluation
        data['Gender'] = data['Gender'].map({'Female': 1, 'Male': 0})
        data['Smoking'] = data['Smoking'].map({'No': 0, 'Yes': 1})
        data['GeneticRisk'] = data['GeneticRisk'].map({'indicating Low': 0, 'indicating Medium': 1 , 'indicating High':2})
        data['CancerHistory'] = data['CancerHistory'].map({'No': 0, 'Yes': 1})
        data['Diagnosis'] = data['Diagnosis'].map({'Negative': 0, 'Positive': 1})
    '''
    st.code(code, language="python")

    st.write("4. ผลลัพท์การเตรียมข้อมูล")
    df = df.dropna()
    df['Gender'] = df['Gender'].map({'Female': 1, 'Male': 0})
    df['Smoking'] = df['Smoking'].map({'No': 0, 'Yes': 1})
    df['GeneticRisk'] = df['GeneticRisk'].map({'indicating Low': 0, 'indicating Medium': 1 , 'indicating High':2})
    df['CancerHistory'] = df['CancerHistory'].map({'No': 0, 'Yes': 1})
    df['Diagnosis'] = df['Diagnosis'].map({'Negative': 0, 'Positive': 1})
    st.dataframe(df)



    st.write("5. ทำการแยก Feature และ Target รวมทั้งเตรียมข้อมูลเป็น Train sets และ Test sets เพื่อนำไปใช้")
    code = '''
        #Separate Features and Target
        X = data.drop('Diagnosis', axis=1)  # Features
        y = data['Diagnosis']  # Target

        # Separate Train sets (80 %) and Test sets (20 %) 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    '''
    st.code(code, language="python")


   


    
    #Train alogorithm
    st.title(":red[Traning] and :orange[Testing]")

    st.subheader("**Algorithm**🎯")
    st.markdown("""
                1. Random Forest : เป็นอัลกอริธึมที่พัฒนาต่อจาก Decision Tree โดยใช้แนวคิด Ensemble Learning หรือการรวมโมเดลหลายตัวเข้าด้วยกันเพื่อลดปัญหา Overfitting และเพิ่มความแม่นยำของผลลัพธ์
                2. SVM (Support Vector Machine) : เป็นอัลกอริธึมที่ใช้สำหรับการจำแนกประเภท (Classification) และการพยากรณ์ค่า (Regression) โดยเน้นการหาขอบเขตที่ดีที่สุดในการแบ่งกลุ่มข้อมูล
              
    """)
    st.subheader("**Random Tree**🎯",divider=True)
    st.write("1. การ train Model ด้วย Random Tree")
    code = '''
        # สร้างโมเดล Random Tree
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
    '''
    st.code(code, language="python")
    st.write("2. ดูผลลัพท์ความแม่นยำในการทำนาย")
    
    code = '''
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
     
        y_pred = model.predict(X_test)

        # คำนวณ Accuracy
        accuracy = accuracy_score(y_test, y_pred)

        #  Classification Report
        report = (classification_report(y_test, y_pred))

        #  Confusion Matrix
        martirx = (confusion_matrix(y_test, y_pred))

    '''
    st.code(code, language="python")

    #SVM train
    st.subheader("SVM (Support Vector Machine)🎯",divider=True)

    st.write("1. การ train Model ด้วย SVM")
    
    code = '''
        from sklearn.svm import SVC

        # สร้างโมเดล SVM
        modelSVM = SVC(kernel='poly', probability=True)  
        modelSVM.fit(X_train, y_train)

    '''
    st.code(code, language="python")
    st.markdown("""
        *เนื่องจากข้อมูลไม่ซับซ้อนมากจึงทำการ Normalization*
              
    """)
    
    st.write("2. ดูผลลัพท์ความแม่นยำในการทำนาย")
    
    code = '''
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    
        y_pred = model.predict(X_test)

        # คำนวณ Accuracy
        accuracy = accuracy_score(y_test, y_pred)

        #  Classification Report
        report = (classification_report(y_test, y_pred))

        #  Confusion Matrix
        martirx = (confusion_matrix(y_test, y_pred))

    '''
    st.code(code, language="python")


   
    








elif selected == "Neural Network Documentation":
    st.title("ℹ️ เกี่ยวกับ")
    st.write("เว็บแอปนี้ใช้ Streamlit ในการพัฒนา")





