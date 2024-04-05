import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.compose import ColumnTransformer
import streamlit as st
import pickle

st.image("https://th.bing.com/th/id/OIP.Bedkg0wAWrtNGo7YrjmVwgHaC4?rs=1&pid=ImgDetMain")
# Title for the web app
st.title("Customer Churn Prediction using ML")

# Initialize an empty list to store user inputs
sample = []
result = None

# Gender input
gender = st.radio('Enter your Gender', ['Male', 'Female'])
sample.append(gender)

# Senior citizen input
senior_citizen = st.radio('Are You a Senior Citizen?', [1, 0], captions=['Yes', 'No'])
sample.append(senior_citizen)

# Partner input
partner = st.radio('Do you have a Partner?', ['Yes', 'No']) 
sample.append(partner)

# Dependents input
dependents = st.checkbox('Do you have Dependents?')
sample.append(dependents)

# Tenure input
tenure = st.number_input("Enter tenure")
sample.append(tenure)

# Phone service input
phone_service = st.radio('Do you have a mobile service?', ['Yes', 'No'])
sample.append(phone_service)

# Multiple lines input
multiple_lines = st.selectbox('Do you have multiple lines?', ('No phone service', 'Yes', 'No'))
sample.append(multiple_lines)

# Internet service input
internet_service = st.selectbox('Type of Internet Service', ('DSL', 'Fiber optic', 'No'))
sample.append(internet_service)

# Online security input
online_security = st.selectbox('Do you have online security?', ('Yes', 'No', 'No internet service'))
sample.append(online_security)

# Online backup input
online_backup = st.selectbox('Do you have online backup?', ('Yes', 'No', 'No internet service'))
sample.append(online_backup)

# Device protection input
device_protection = st.selectbox('Do you have device protection?', ('Yes', 'No', 'No internet service'))
sample.append(device_protection)

# Tech support input
tech_support = st.selectbox('Do you have tech support?', ('Yes', 'No', 'No internet service'))
sample.append(tech_support)

# Streaming TV input
streaming_tv = st.selectbox('Do you have streaming TV?', ('Yes', 'No', 'No internet service'))
sample.append(streaming_tv)

# Streaming movies input
streaming_movies = st.selectbox('Do you have streaming movies?', ('Yes', 'No', 'No internet service'))
sample.append(streaming_movies)

# Contract input
contract = st.selectbox('Type of Contract', ('Month-to-month', 'One year', 'Two year'))
sample.append(contract)

# Paperless billing input
paperless_billing = st.radio('Do you have paperless billing?', ['Yes', 'No'])
sample.append(paperless_billing)

# Payment method input
payment_method = st.selectbox('Method of Payment', ('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'))
sample.append(payment_method)

# Monthly charges input
monthly_charges = st.number_input("Enter monthly charges")
sample.append(monthly_charges)

# Total charges input
total_charges = st.number_input("Enter total charges")
sample.append(total_charges)

# Load the trained model
with open("customer_churn_model.pkl", "rb") as f:
    model = pickle.load(f)

# Predict the result when the Submit button is clicked
if st.button("Submit") == True:
    # Create a DataFrame from the user inputs
    result_df = pd.DataFrame(np.array([sample]).reshape(1,-1), columns=['gender','SeniorCitizen','Partner', 'Dependents', 'tenure','PhoneService', 'MultipleLines', 
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 
        'PaymentMethod','MonthlyCharges', 'TotalCharges'])
# Predict the churn result
    result = model.predict(result_df)[0]
else:
    pass

# Display the prediction result
if result == "Yes":
    st.subheader(":red[The customer churned]")
    st.image("https://th.bing.com/th/id/OIP.zzndj0e06wI9eh_R9IeifgHaEK?rs=1&pid=ImgDetMain")
elif result == "No":
    st.subheader(":green[The customer didn't churn]")
    st.image("https://th.bing.com/th/id/OIP._Cmky23BQSWmXgY62E_41gHaG5?w=960&h=895&rs=1&pid=ImgDetMain")

