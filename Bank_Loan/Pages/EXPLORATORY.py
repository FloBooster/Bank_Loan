# 0. Importing Libraries**
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import LabelEncoder
from ydata_profiling import ProfileReport
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import joblib
import streamlit as st
from PIL import Image
import webbrowser
def main():
  st.title("EXPLORATORY")
  st.title("Loan Bank Prediction App")
  st.sidebar.title("Loan Application Form")

# 1. Reading Data
  @st.cache_data
  def load_data():
    data= pd.read_csv("Loan_Bank.csv")
    return data
  df = load_data()
  df_sample = df.sample(100)
  if st.sidebar.checkbox("Show The Dataset",False):
    st.subheader("Echantillon de 100 observations")
    st.write(df_sample)
   
# 2. Data Preprocessing
  info = df.info()
  info = pd.DataFrame(info)
  vu = info.head()
  stat_num = df.describe()
  stat_obj = df.describe(include = "object")

  st.sidebar.subheader("Data Overview")
  if st.sidebar.checkbox("Data Info",False):
    st.subheader("Data Info")
    st.write(vu)
  if st.sidebar.checkbox("Numerical Variables Describe",False):
    st.subheader("Numerical Describe")
    st.write(stat_num)
  if st.sidebar.checkbox("Object Variables Describe",False):
    st.subheader("Object Variables Describe")
    st.write(stat_obj)

  #Handle Missing Values
  missing_values = df.isnull().sum()
  if st.sidebar.checkbox("Missing Values",False):
    st.subheader("Missing Values")
    st.write(missing_values)

  #Splittin Dataset into Categorical and numerical variables
  cat_data =[]
  num_data =[]
  for i,c in enumerate (df.dtypes):
    if c==object:
      cat_data.append(df.iloc[:,i])
    else:
      num_data.append(df.iloc[:,i])
  cat_data = pd.DataFrame(cat_data).transpose()
  num_data = pd.DataFrame(num_data).transpose()

  cat_data = cat_data.apply(lambda x: x.fillna(x.value_counts().index[0]))
  cat_data.isnull().sum().any()

  num_data.fillna(method="bfill", inplace=True)
  num_data.isnull().sum().any()

    #Target Encoding
  target_value = {"Y":1,"N":0}
  target = cat_data["Loan_Status"]
  cat_data.drop("Loan_Status",axis = 1,inplace = True)
  target = target.map(target_value)

    #Encoding Others categorical variables
  Le = LabelEncoder()
  for i in cat_data:
    cat_data[i] = Le.fit_transform(cat_data[i])

    #Drop Loan_ID Column
  cat_data.drop("Loan_ID",axis = 1,inplace = True)

    #Define Features and Target Dataframes
  X =pd.concat([num_data,cat_data],axis = 1)
  y = target

  # 3. Exploratory Data Analysis**
    #Concact cat_data and num_data to obtain new_data
  data = pd.concat([num_data,cat_data,target],axis = 1)

    #Visualization of the target variable according to categorical variables
      # Create a figure with 3 rows and 2 columns of subplots
  st.sidebar.subheader("Data Visualization")
  def plot_categorical(data):
    fig_1, axes = plt.subplots(3, 2, figsize=(30, 26))

      # Plot Loan_Status vs Gender
    sns.countplot(ax=axes[0, 0], x="Gender", hue="Loan_Status", data=data)
    axes[0, 0].set_title("Loan Status by Gender")

      # Plot Loan_Status vs Married
    sns.countplot(ax=axes[0, 1], x="Married", hue="Loan_Status", data=data)
    axes[0, 1].set_title("Loan Status by Married")

      # Plot Loan_Status vs Dependents
    sns.countplot(ax=axes[1, 0], x="Dependents", hue="Loan_Status", data=data)
    axes[1, 0].set_title("Loan Status by Dependents")

      # Plot Loan_Status vs Education
    sns.countplot(ax=axes[1, 1], x="Education", hue="Loan_Status", data=data)
    axes[1, 1].set_title("Loan Status by Education")

      # Plot Loan_Status vs Self_Employed
    sns.countplot(ax=axes[2, 0], x="Self_Employed", hue="Loan_Status", data=data)
    axes[2, 0].set_title("Loan Status by Self-Employed")

      # Plot Loan_Status vs Property_Area
    sns.countplot(ax=axes[2, 1], x="Property_Area", hue="Loan_Status", data=data)
    axes[2, 1].set_title("Loan Status by Property Area")

      # Save the figure
    fig_1.savefig("categorical_data.png")
    return "categorical_data.png"
  if st.sidebar.checkbox("Categorical Graph", False):
   saved_fig_1 = plot_categorical (data)
   st.subheader("Categorical Graph Visualization")
   st.image(saved_fig_1, caption="Categorical Graph Visualization", use_column_width=True)
   st.subheader("Comments")
   st.markdown("""
    * We observe that men are more likely to obtain a loan than women.
    * Married people are more likely to get a loan than single people.
    * The number of dependents has no significant impact on obtaining a loan.
    * People with a university degree are more likely to get a loan than those with a lower degree.
    * Self-employed workers are less likely to obtain a loan than employees.
    * People living in an urban area are more likely to get a loan than those living in a rural area.
    """)
  #Visualization of the target variable based on numeric variables***
   # Create a figure with 2 rows and 2 columns of subplots
  
  def plot_numerical(data):
    fig_2, axes = plt.subplots(2, 2, figsize=(30, 26))

          # Plot Loan_Status vs ApplicantIncome
    sns.boxplot(ax=axes[0, 0], x="Loan_Status", y="ApplicantIncome", data=data)
    axes[0, 0].set_title("Loan Status by Applicant Income")

          # Plot Loan_Status vs CoapplicantIncome
    sns.boxplot(ax=axes[0, 1], x="Loan_Status", y="CoapplicantIncome", data=data)
    axes[0, 1].set_title("Loan Status by Coapplicant Income")

          # Plot Loan_Status vs LoanAmount
    sns.boxplot(ax=axes[1, 0], x="Loan_Status", y="LoanAmount", data=data)
    axes[1, 0].set_title("Loan Status by Loan Amount")

          # Plot Loan_Status vs Credit_History
    sns.countplot(ax=axes[1, 1], x="Credit_History", hue="Loan_Status", data=data)
    axes[1, 1].set_title("Loan Status by Credit History")
 
        # Save the figure
    fig_2.savefig("numerical_data.png")
    return "numerical_data.png"
  if st.sidebar.checkbox("Numerical Graph", False):
   saved_fig_2 = plot_numerical(data)
   st.subheader("Numerical Graph Visualization")
   st.image(saved_fig_2, caption="Numerical Graph Visualization", use_column_width=True)
   st.subheader("Comments")
   st.markdown("""
    * People with higher applicant income are more likely to get a loan.
    * People with higher co-applicant income are more likely to get a loan.
    * People with a higher loan amount are less likely to get a loan.
    * People with a credit history are more likely to get a loan.
    """)
   
  # Data Profiling Report
  # GENERATION DU RAPPORT
  st.sidebar.subheader("Data Profiling Report")
  profile = st.sidebar.button("Profiling Report")  
  if profile:
    prof = ProfileReport(data)
    prof.to_file(output_file="report_loan.html") 

# TELECHARGEMENT DU RAPPORT
    telechargement = st.sidebar.download_button("Open the Report","report_loan.html")
    if telechargement:
      webbrowser.open("report_loan.html")

if __name__ == '__main__':
   
  main()



