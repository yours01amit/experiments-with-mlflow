import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns


mlflow.set_tracking_uri('http://127.0.0.1:5000')

# Load the wine dataset
wine = load_wine()

X = wine.data 
y = wine.target

# Train test split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=20,random_state=42)

# Define the params for Random Forest model
max_depth = 40
n_est = 10

# Men
mlflow.set_experiment('myexperimets1')

with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=max_depth,n_estimators=n_est,random_state=42)
    rf.fit(X_train,y_train)

    y_pred = rf.predict(X_test)

    accuracy = accuracy_score(y_test,y_pred)

    mlflow.log_metric('accuracy',accuracy)
    mlflow.log_param('max_depth',max_depth)
    mlflow.log_param('estimators',n_est)

    # Creating a confusion matrix plot
    metrix = confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(metrix,annot=True,fmt='d',cmap='Blues',xticklabels=wine.target_names,yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('predicted')
    plt.title('Confusion Matrix')

    # save plot
    plt.savefig('confusion-matrix.png')

    # log artifacts using mlflow
    mlflow.log_artifact('confusion-matrix.png')
    mlflow.log_artifact(__file__)

    print(accuracy)
