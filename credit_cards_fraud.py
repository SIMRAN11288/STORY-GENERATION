import kagglehub
import pandas as pd
import numpy as np
import streamlit as st
import os
def load_model():
    # Download latest version
    path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    #print("Path to dataset files:", path)
    print(os.listdir(path))
    file_path=os.path.join(path,'creditcard.csv')
    data=pd.read_csv(file_path)
    print(data.columns)
    data=data.iloc[:10000,:]
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split
    X=data.drop(columns=['Class'])
    iso=IsolationForest(contamination=0.01,random_state=42)
    y_pred_iso=iso.fit_predict(X)
    data['IsoForest_Fraud'] = np.where(y_pred_iso == -1, 1, 0)
    lof=LocalOutlierFactor(n_neighbors=20,contamination=0.01)
    y_pred_lof=lof.fit_predict(X)
    data['LocalOutlierFactor'] = np.where(y_pred_lof == -1, 1, 0)
    #columns
    features=X.columns
    Y=data['Class']
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
    classifier=XGBClassifier()
    classifier.fit(X_train,Y_train)
    return classifier,features
classifier,features=load_model()
#straemlit UI
st.title("CREDIT CARD FRAUD DETECTION")
st.write("enter values")
#enter values
time_val=st.number_input("Enter transaction time (seconds since first transaction, taking transaction time from 9:00AM): ")
amount=st.number_input("Enter transaction amount: ",min_value=0.0, format="%.2f")
if st.button('predict fraud'):
    # Create dummy input (fill V1..V28 with zeros)
    input=pd.DataFrame([[time_val]+[0]*28 + [amount]],columns=features)
    
    #Prediction
    prediction=classifier.predict(input)[0]
    
    if prediction==1:
        st.error("fraudulent Transaction")
    else:
        st.success("Legitimate transaction")
    

