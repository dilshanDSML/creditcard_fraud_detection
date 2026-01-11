import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

data=pd.read_csv('creditcard.csv')
legit=data[data["Class"]==0]
fraud=data[data["Class"]==1]
legit_sample=legit.sample(n=len(fraud),random_state=2)
data=pd.concat([legit_sample,fraud],axis=0)
X=data.drop(columns="Class",axis=1)
Y=data["Class"]
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)
model=LogisticRegression()
model.fit(X_train,Y_train)
y_pred=model.predict(X_train)
#web app
st.title("Credit Card Fraud Detection")
input_df=st.text_input("Enter all required features values")
input_df_splited=input_df.split(',')
submit=st.button('Predict')
if submit:
    features=np.asarray(input_df_splited,dtype=np.float64)
    prediction=model.predict(features.reshape(1,-1))
    if(prediction[0]==0):
        st.write("The transaction is Legitimate")
    else:
        st.write("The transaction is Fraudulent")