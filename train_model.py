import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import joblib


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report

data=pd.read_csv("kyphosis.csv")
print(data)
print("NULL_values:", data.isnull().sum())
print("columns:",data.columns)
print(data.describe())

#defining x and y variables and slit the data
X=data.drop("Kyphosis",axis=1)
y=data["Kyphosis"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

mlflow.set_experiment("Kyphosis_experiment")
with mlflow.start_run():
         #model=LogisticRegression()
          model=DecisionTreeClassifier(max_depth=4, class_weight="balanced", random_state=42)
         # model=RandomForestClassifier(n_estimators=200)
          learn_data=model.fit(X_train,y_train)
          y_predcs=model.predict(X_test)

          accuracy=accuracy_score(y_test,y_predcs)
          classification=classification_report(y_test,y_predcs)
         
          mlflow.log_metric("accuracy", accuracy)
          with open("classification_report.txt", "w") as f:
              f.write(classification)
          mlflow.log_artifact("classification_report.txt")

          mlflow.sklearn.log_model(sk_model=model, name="model", registered_model_name="Kyphosis_Model")
          print("Accuracy_Score:", accuracy)

          
joblib.dump(model, "Kyphosis.pkl")
print("✅ Model saved as Kyphosis.pkl")


