# 0. Importing Libraries**
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import joblib
from sklearn.preprocessing import LabelEncoder
from ydata_profiling import ProfileReport
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import streamlit as st


def main():
  st.title("PREDICTION")
  st.title("Loan Bank Prediction App")
  st.sidebar.title("Loan Application Form")  
 
 # Load Logistic Regression Model 
  st.subheader("Logistics Regression Model")
  if st.sidebar.checkbox("Load Model", False):
    def loading_model():
      # Load the model from the picke file
      with open('Logistic Regression.pkl', 'rb') as file:
          model = joblib.load(file)
      return model
    st.success("Model loaded successfully.")
    

# Create inputs for the variables
  st.sidebar.subheader("Enter Applicant Details")
  st.sidebar.markdown("""
   1. Gender (Male, Female)             
   2. Education (Graduate, Not Graduate)                  
   3. Self Employed (Yes, No)
   4. Applicant Income (minimum: 5000)
   5. Coapplicant Income (minimum: 5000)
   6. Loan Amount (minimum: 0) 
   7. Credit History (1 for Yes, 0 for No) 
   """)
   # Predict the loan status using the loaded model

  if st.sidebar.checkbox("Enter inputs"): 
   Gender = st.selectbox("Gender", ["Male", "Female"])
   Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
   Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])
   ApplicantIncome = st.number_input("ApplicantIncome", min_value=5000,step=1000)
   CoapplicantIncome = st.number_input("CoapplicantIncome", min_value=5000,step=500)
   LoanAmount = st.number_input("LoanAmount", min_value=0,step= 1000)
   Credit_History = st.selectbox("Credit History", [1.0, 0.0])
   
   # Convert categorical variables to numerical
   Gender = 1 if Gender == "Male" else 0
   Education = 1 if Education == "Graduate" else 0
   Self_Employed = 1 if Self_Employed == "Yes" else 0
   
   # Create a prediction button
   input_data ={"Gender": Gender,
   "Education": Education,
   "Self_Employed": Self_Employed,
   "ApplicantIncome": ApplicantIncome,
   "CoapplicantIncome": CoapplicantIncome,
   "LoanAmount" : LoanAmount, 
   "Credit_History" : Credit_History, 
  
   } 
 
         # Create a dataframe with the input data
   features = pd.DataFrame(input_data,index =[0])
   if st.button("Predict Loan Status"):
    model = loading_model()
    prediction = model.predict(features)
    if prediction == 1:
     st.success("Approved")
    else:
      st.error("Not Approved")

  
if __name__ == '__main__':
    main()
