import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
# from sklearn.neighbors import KNeighborsClassifier
import joblib

data = pd. read_csv("Dataset\cancer_dataset.csv")
data.head()


# drop null value
data = data.dropna()

# Convert data to number
data['Gender'] = data['Gender'].map({'Female': 1, 'Male': 0})
data['Smoking'] = data['Smoking'].map({'No': 0, 'Yes': 1})
data['GeneticRisk'] = data['GeneticRisk'].map({'indicating Low': 0, 'indicating Medium': 1 , 'indicating High':2})
data['CancerHistory'] = data['CancerHistory'].map({'No': 0, 'Yes': 1})
data['Diagnosis'] = data['Diagnosis'].map({'Negative': 0, 'Positive': 1})


# Train
X = data.drop('Diagnosis', axis=1)  # Features
y = data['Diagnosis']  # Target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)



filename = 'randomForest_model.sav'
joblib.dump(model, filename)


y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# คำนวณ Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# แสดง Classification Report
print(classification_report(y_test, y_pred))

# แสดง Confusion Matrix
print(confusion_matrix(y_test, y_pred))