import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split







from django.shortcuts import render
# import joblib
# model=joblib.load('./savedModels/model.joblib')
# # import joblib

# # try:
# #     model = joblib.load('model.joblib')
# # except FileNotFoundError:
# #     print("The 'model.joblib' file was not found.")
# #model = load('./savedModels/model.joblib')

def predictor(request):
    return render(request, 'main1.html')

def formInfo(request):
    age= request.GET['age']
    workclass= request.GET['workclass']
    # fnlwgt= request.GET['fnlwgt']
    education= request.GET['education']
    marital_status= request.GET['marital-status']
    occupation= request.GET['occupation']
    sex= request.GET['sex']
    capital_gain= request.GET['capital-gain']
    capital_loss= request.GET['capital-loss']
    hours_per_week= request.GET['hours-per-week']
    native_country= request.GET['native-country']

    data=pd.read_csv('C:/Users/Dell/Desktop/ML_project/classification/notebooks/income_evaluation.csv')
    data
    data.shape
    data.isna().sum()
    data.head()
    data.columns
    data=data.drop([' relationship',' education-num',' fnlwgt',' race'],axis=1)
    data
    data[' workclass'] = data[' workclass'].apply(lambda x:x.strip( ))
    data[' education'] = data[' education'].apply(lambda x:x.strip( ))
    data[' marital-status'] = data[' marital-status'].apply(lambda x:x.strip( ))
    data[' occupation'] = data[' occupation'].apply(lambda x:x.strip( ))
    data[' native-country'] = data[' native-country'].apply(lambda x:x.strip( ))
    data[' sex'] = data[' sex'].apply(lambda x:x.strip( )) 
    x=data.drop(' income',axis=1)
    y=data[' income']
##label encoder
    le_workclass=LabelEncoder()
    x[' workclass']=le_workclass.fit_transform(x[' workclass'])

    le_education=LabelEncoder()
    x[' education']=le_education.fit_transform(x[' education'])

    le_occupation=LabelEncoder()
    x[' occupation']=le_occupation.fit_transform(x[' occupation'])


    le_sex=LabelEncoder()
    x[' sex']=le_sex.fit_transform(x[' sex'])

    le_native_country=LabelEncoder()
    x[' native-country']=le_native_country.fit_transform(x[' native-country'])


    le_marital_status=LabelEncoder()
    x[' marital-status']=le_marital_status.fit_transform(x[' marital-status'])
    x
    y
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=243)
    model=DecisionTreeClassifier()
    model.fit(x_train,y_train)
    model.score(x_test,y_test)
    model.get_depth()
    model1=DecisionTreeClassifier(max_depth=11) #()hyperparameter(used for better model)
    model1.fit(x_train,y_train)
    model1.score(x_test,y_test)
    model1.get_depth()
    y_pred=model1.predict(x)
    pd.DataFrame({'True values':y,"Predictions":y_pred})
    x.columns

    y_pred=model1.predict([[age,workclass,education,marital_status,occupation,sex,capital_gain,capital_loss,hours_per_week,native_country]])
    y_pred
    y_pred=model1.predict(x_test)

    salary=y_pred[0]
    
    
    return render(request, 'result1.html',{'result':salary})