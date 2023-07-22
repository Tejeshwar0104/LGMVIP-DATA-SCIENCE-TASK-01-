# LGMVIP-DATA-SCIENCE-TASK-01

Code:

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

datas = pd.read_csv("/content/Iris.csv")
datas

datas.isnull().sum()

datas.describe()

datas.plot(kind="scatter",x="SepalLengthCm",y="SepalWidthCm")

X = datas[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
Y = datas['Species']

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.25,random_state = 50)

classifier = GaussianNB()
classifier.fit(X_train,Y_train)

pred = classifier.predict(X_test)

accuracy = accuracy_score(Y_test,pred)
print("The Accuracy of the model is: ",accuracy)
