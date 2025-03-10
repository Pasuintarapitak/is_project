import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np
import joblib



data = pd. read_csv("Dataset/cancer_dataset.csv")

data = data.dropna()
# แปลงข้อมูลเป็นตัวเลข
data['Gender'] = data['Gender'].map({'Female': 1, 'Male': 0})
data['Smoking'] = data['Smoking'].map({'No': 0, 'Yes': 1})
data['GeneticRisk'] = data['GeneticRisk'].map({'indicating Low': 0, 'indicating Medium': 1 , 'indicating High':2})
data['CancerHistory'] = data['CancerHistory'].map({'No': 0, 'Yes': 1})
data['Diagnosis'] = data['Diagnosis'].map({'Negative': 0, 'Positive': 1})


X = data.drop('Diagnosis', axis=1)  # Features
y = data['Diagnosis']  # Target
# แบ่งข้อมูลเป็น Train และ Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# สร้างโมเดล SVM
model = SVC(kernel='linear', probability=True)  # kernel สามารถเปลี่ยนเป็น 'rbf', 'poly' ได้
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# คำนวณ Accuracy
accuracy = accuracy_score(y_test, y_pred)

#  Classification Report
report = (classification_report(y_test, y_pred))

#  Confusion Matrix
martirx = (confusion_matrix(y_test, y_pred))

filename = 'svm_model.sav'
joblib.dump(model, filename)