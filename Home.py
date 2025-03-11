from pyexpat import model

from sklearn.metrics import classification_report, confusion_matrix
import streamlit as st

import random
import pandas as pd

from streamlit_option_menu import option_menu
from model_handler import predict  # ดึงฟังก์ชันพยากรณ์
import tensorflow_datasets as tfds


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


    st.title(":orange[Dog] Breed Prediction Model 🐶")
    st.write("โมเดลทำนายสายพันธุ์สุนัขจากรูปภาพ")

    def load_dataset():
        dataset_name = "stanford_dogs"
        dataset, info = tfds.load(dataset_name, with_info=True, as_supervised=True)
        return dataset, info

    dataset, info = load_dataset()
    label_names = info.features["label"].int2str 

    train_data, test_data = dataset["train"], dataset["test"]
    sample_data = list(train_data.take(5))  
    #image
    cols = st.columns(5)  # สร้าง columns 5 อัน
    for col_idx, (image, label) in enumerate(sample_data):
        breed_name = label_names(label.numpy()) 
        with cols[col_idx]:  
            st.image(image.numpy(), caption=breed_name, width=180) 
    
    st.markdown("***การทำนายสายพันธ์ุสุนัข 120 สายพันธู์ โดยการวิเคราะห์คุณลักษณะต่างๆจากรูปภาพ***")
    st.markdown("***Raw Dateset : https://www.tensorflow.org/datasets/catalog/stanford_dogs***")
    
    #prepare data
    st.title(":red[Prepare] dataset")

    st.write("1. ทำการดึง dataset  'stanford_dogs ' จาก tensorflow และแยก Train set กับ Test set")
    code = '''
    def load_dataset():
        dataset_name = "stanford_dogs"
        dataset, info = tfds.load(dataset_name, with_info=True, as_supervised=True)
        return dataset, info
    dataset, info = load_dataset() 

    train_data, test_data = dataset["train"], dataset["test"]
    '''
    st.code(code, language="python")
    st.markdown("***standford_dogs มีภาพทั้งหมด 20,580 รูป : Train 12,000 รูป + Test 8,580 รูป***")

    st.write("2. ปรับขนาดรูปภาพ และ batch size")
    code = '''
        batch_size = 32
        image_size = (128, 128)  
    '''
    st.code(code, language="python")


    st.write("3. ปรับแต่งรูปภาพ และขนาดเพื่อเตรียมนำไป train")
    code = '''
        def preprocess(image, label):
            image = tf.cast(image, tf.float32) / 255.0  # Normalize
            image = tf.image.resize(image, image_size)  # Resize
            return image, label
    '''
    st.code(code, language="python")

    st.write("4. ข้อมูล Train และ ข้อมูล Test")
    code = '''
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

    '''
    st.code(code, language="python")

    #Train alogorithm
    st.title(":red[Traning] and :orange[Testing]")

    st.subheader("**Algorithm**🎯")
    st.markdown("""
        - Convolutional Neural Network (CNN) : เป็นโครงสร้าง Neural Network ที่เป็นที่นิยมตัวนึง ซึ่งเหมาะกับการจำแนกรูปของข้อมูลประเภทรูปภาพ โดยวิเคราะห์จาก ขอบภาพ พื้นผิว รูปทรง วัตถุในภาพ
        โดยมี Convolutional Layers สำหรับ features จากภาพ เรียนรู้จากภาพ 
                      
    """)
    st.write("**CNN ประกอบด้วย**")
    st.markdown("""
  
        1. Convolutional Layer : ดึง features จากภาพโดยใช้ Filter(Kernel) ที่เลื่อนผ่านและคำนวณค่า
        2. Pooling Layer : ลดขนาดเพื่อเพิ่มประสิทธิภาพของโมเดล
        3. Flatten Layer : แปลงข้อมูลให้สามารถนำไปใช้ได้ (vector)
        4. Fully Connected Layer : เชื่อมต่อขข้อมูลทั้งหมด เพื่อทำการทำนายผลลัพท์
        5. Activation Function : เพิ่มความสามารถในการเรียนรู้ เช่น ReLU และ Softmax   
    """)
    st.markdown("""
        - MoblieNet : เป็นโครงสร้าง CNN ที่ Google คิดค้น ที่มีช่วยให้ลดขนาด model และเพิ่มความเร็ว และมีความแม่นยำสูง เพื่อให้ทำงานได้ดีบน Moblie Devices              
    """)
    st.write("**โครงสร้าง MoblieNet**")
    st.markdown("""
        1. Depthwise Separable Convolution : เนื่องจาก CNN ใช้ Convolutional Layers ที่ต้องตำนวณเยอะมาก สิ่งนี้จะช่วยเพื่อลดพารามิเตอร์  
        2. Width Multiplier : ทำให้ model ขนาดเล็กลง โดยลดจำนวน feature map
        3. Resolution Multiplier : ลดขนาด input เพื่อเพิ่มความเร็ว    
        """)
    st.subheader("**Training model**🎯",divider=True)
    st.write("1. โหลด MoblieNetV2 โดยไม่ให้โมเดลปรับพารามิเตอร์ที่ฝึกระหว่างฝึก")
    code = '''
        base_model = tf.keras.applications.MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights="imagenet")
        base_model.trainable = False
    '''
    st.code(code, language="python")

    st.write("2. train model โดยใช้ MoblieNetV2")
    code = '''
        #สร้าง Model
        model = tf.keras.Sequential([
            base,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(120, activation="softmax")
        ])

        model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"]) #compile model
        #adam เป็นอัลกอริทึมที่ช่วยในการปรับ learning rate ได้ดี

        class_weights = {i: 1.0 for i in range(120)} # 120 breeds
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
        model.fit(train_data, epochs=10, validation_data=test_data, class_weight=class_weights, callbacks=[early_stopping])

        model.save('MobileNetV2_model.keras')  # savemodel
        return model
    '''
    st.code(code, language="python")

    st.write("Train function")
    code = '''
    def train_model():
        base = tf.keras.applications.MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights="imagenet")
        base.trainable = False

        #สร้าง Model
        model = tf.keras.Sequential([
            base,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(120, activation="softmax")
        ])

        model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"]) #compile model
        #adam เป็นอัลกอริทึมที่ช่วยในการปรับ learning rate ได้ดี

        class_weights = {i: 1.0 for i in range(120)} # 120 breeds
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
        model.fit(train_data, epochs=10, validation_data=test_data, class_weight=class_weights, callbacks=[early_stopping])

        model.save('MobileNetV2_model.keras')  # save model
        return model
    '''
    st.code(code, language="python")

    #Train alogorithm
    st.title(":red[Neural] Network :orange[App] ")

    st.write("1. โหลด model จากไฟล์ .keras")
    code = '''
        def load_model():
            return tf.keras.models.load_model('models/MobileNetV2_model.keras')

        model = load_model()
    '''
    st.code(code, language="python")

    
    st.write("2. โหลด dataset เพื่อใช้ตอนทำนาย")
    code = '''
        def get_label_map():
            _, info = tfds.load("stanford_dogs", with_info=True)
            return info.features['label'].int2str  # convert index to breed

        label_map = get_label_map()
    '''
    st.code(code, language="python")

    st.write("3. แปลงรูปภาพเพื่อใช้ทำนาย")
    code = '''
        def preprocess_image(image):
            image = image.convert("RGB")  
            image = image.resize((128, 128))  
            image = np.array(image) / 255.0  
            image = np.expand_dims(image, axis=0)  
            return image

    '''
    st.code(code, language="python")

    st.write("4. อัพโหลดรูปภาพและทำนายผลลัพท์")
    code = '''
        uploaded_file = st.file_uploader("Upload your dog picture", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="📸 ภาพที่อัปโหลด", use_column_width=True)

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

    '''
    st.code(code, language="python")