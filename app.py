#!/usr/bin/env python
# coding: utf-8

# In[6]:


import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from category_encoders import BinaryEncoder, TargetEncoder
import joblib

model = joblib.load('trained_model.pkl')
t_encoder = joblib.load('T_encoder.pkl')
b_encoder = joblib.load('B_encoder.pkl')
s_scaler = joblib.load('S_scaler.pkl')
selected_columns = joblib.load('selected_columns.pkl')

num_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
bin_cat_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
cat_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod'] 

st.set_page_config(page_title="📞 Telco Churn Prediction", layout="centered")
st.title("📞 Telco Customer Churn Predictor")
st.markdown("Predict if a customer is likely to churn based on input features.")

gender = st.selectbox('Gender', ['Male', 'Female'])
senior_citizen = st.selectbox('Senior Citizen', [0, 1])
partner = st.selectbox('Partner', ['Yes', 'No'])
dependents = st.selectbox('Dependent', ['Yes', 'No'])
tenure = st.number_input('tenure', min_value=1, max_value=100)
phone_service = st.selectbox('Phone Service', ['Yes', 'No'])
MultipleLines = st.selectbox('Multiple Lines', ['Yes', 'No', 'No phone service'])
InternetService = st.selectbox('Internet Service', ['Fiber optic', 'DSL', 'No'])
OnlineSecurity  = st.selectbox('Online Security', ['No', 'Yes', 'No internet service'])
OnlineBackup  = st.selectbox('Online Backup', ['No', 'Yes', 'No internet service'])
DeviceProtection = st.selectbox('Device protection', ['No', 'Yes', 'No internet service']) 
TechSupport = st.selectbox('Tech support', ['No', 'Yes', 'No internet service'])
StreamingTV = st.selectbox('Streaming TV', ['No', 'Yes', 'No internet service'])
StreamingMovies = st.selectbox('Streaming Movie', ['No', 'Yes', 'No internet service'])
Contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
PaperlessBilling = st.selectbox('Paperless billing', ['No', 'Yes'])
PaymentMethod = st.selectbox('PaymentMethod', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])  
MonthlyCharges = st.number_input('Monthly charges', min_value=19.0, max_value=21.0)
TotalCharges = st.number_input('Total charges', min_value=15, max_value=100) 

def encoder(inputs):
    
    X_t = t_encoder.transform(inputs[cat_cols])
    X_b = b_encoder.transform(inputs[bin_cat_cols])
    X_s = pd.DataFrame(
        s_scaler.transform(inputs[num_cols]),
        columns=num_cols
    )

    X_test = pd.concat([
        pd.DataFrame(X_t),
        pd.DataFrame(X_b),
        pd.DataFrame(X_s)
    ], axis=1)

    X_test_new = X_test.reindex(columns=selected_columns)
    
    pre = model.predict(X_test_new)

    if pre[0] == 0:
        st.success("🎉 Customer will NOT churn!")
    else:
        st.error("⚠️ Customer is likely to CHURN!")


input_dict = {}

if st.button("Predict Churn") :
    input_dict = {
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': MultipleLines,
        'InternetService': InternetService,
        'OnlineSecurity': OnlineSecurity,
        'OnlineBackup': OnlineBackup,
        'DeviceProtection': DeviceProtection,
        'TechSupport': TechSupport,
        'StreamingTV': StreamingTV,
        'StreamingMovies': StreamingMovies,
        'Contract': Contract,
        'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges
        }
    inputs = pd.DataFrame([input_dict])
    encoder(inputs)




