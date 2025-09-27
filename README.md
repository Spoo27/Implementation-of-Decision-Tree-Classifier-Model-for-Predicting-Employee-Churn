# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.

2. Upload and read the dataset.

3. Check for any null values using the isnull() function.

4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.


## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Spoorthi
RegisterNumber:  212224230271
*/
```
```
import pandas as pd
data = pd.read_csv("Employee (1).csv")
data.head()

data.info()
data.isnull().sum()
data['left'].value_counts()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data['salary'] = le.fit_transform(data['salary'])
data.head()

x=data[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']]
x.head()

y=data['left']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state =100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train,y_train)
y_predict=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_predict)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plot_tree(dt, feature_names=x.columns, class_names=['salary', 'left'], filled=True)
plt.show()
```
## Output:
## DATA HEAD:
<img width="1631" height="258" alt="Screenshot 2025-09-27 134406" src="https://github.com/user-attachments/assets/79771928-d549-40b5-966c-acb676d6e4e8" />

## DATASET INFO:
<img width="611" height="598" alt="Screenshot 2025-09-27 134417" src="https://github.com/user-attachments/assets/0b91dcbb-428f-4d49-b95f-1609b2aa19e8" />

## DATASET TRANSFORMED HEAD:
<img width="1618" height="250" alt="Screenshot 2025-09-27 134435" src="https://github.com/user-attachments/assets/26dcc770-1abd-499f-84c2-eccee24b2b47" />

## Accuracy


<img width="400" height="50" alt="Screenshot 2025-09-27 134558" src="https://github.com/user-attachments/assets/b50de99f-a794-4418-8b8e-aece9cf72a66" />


## DATA PREDICTION:
<img width="429" height="49" alt="Screenshot 2025-09-27 134605" src="https://github.com/user-attachments/assets/5a8fe411-7504-421e-a031-8fb542d1335e" />


<img width="1033" height="619" alt="Screenshot 2025-09-27 134925" src="https://github.com/user-attachments/assets/7d577815-05c3-4ca8-976f-221de1754b3e" />


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
