# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:21:42 2020

@author: Muhammad Ali
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math


titanic_data = pd.read_csv("")
titanic_data.head(10)

print("# of pssengers in original data:" +str(len(titanic_data.indix)))


#Analyzing Data
sns.countplotlib("x = Survived", data = titanic_data)
sns.countplotlib("x = Survived", hue ="Sex",data= titanic_data)
sns.countplotlib("x = Survived", hue="Pclass",data = titanic_data)

titanic_data["Age"].plot.hist()
titanic_data["Fare"].plot.hist(bin = 20,figsize=(10,5))  

# informtion
titanic_data.info()
sns.countplot(x="SibSp",data=titanic_data)

#Data Wrangling
titanic_data.isnull()

titanic_data.isnull().sum()
sns.heatmap(titanic_data.isnull(),yticklabels==False, cmap="viridis")
sns.boxplot(x="pcalss", y="Age",titanic_data)

titanic_data.head(5)

titanic_data("Cabin",axis=1,inplace=True)
titanic_data.bead(5)


titanic_data.dropna(inplace=True)
sns.headtmap(titanic_data.isnull(),yticklabels=False, cbar=False)

titanic_data.isnull()

titanic_data.head(2)

pd.get_dunmies(titanic_data['Sex'],drop_first=True)
sex.head(5)

embark=pd.get_dunmies(titanic_data["Embark"],drop_first=True)
embark.head(5)

Pcl=pd.get_dunmies(titanic_data["Pclass"],drop_first=True)
Pcl.head(5)

titanic_data = pd.concat([titanic_data,sex,embark,Pcl],axis=1)
titanic_data,head(5)

titanic_data.drop(['Sex','Embrked','PassengerId','Name','ticket'],axis=1,inplace=True)

titanic_data.drop('Pclass',axis=1,inplace=True)


#Train Data
x = titanic_data.drop("Survived",axis=1)
y =titanic_data['Survived]

from sklearn.cross_validation import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y test_size=0.33, random_state=1)

from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression
logmodel = fit(x_train,y_train)

predictions =logmodel.predict(x_test)
from sklearn.metrics import classification_report

classification_report(y_test, predictions)

from sklearn.matrics(y_test, predictions)
confusion_matrix(y_predictions)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, predictions)
         