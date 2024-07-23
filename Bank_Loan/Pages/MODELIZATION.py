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
def main():
  st.title("MODELIZATION")
  st.title("Loan Bank Prediction App")
  st.sidebar.title("Loan Application Form")

# 1. Reading Data
  @st.cache_data
  def load_data():
    data= pd.read_csv("Loan_Bank.csv")
    return data
  df = load_data()
  df_sample = df.sample(100)
  if st.sidebar.checkbox("Show the Dataset",False):
    st.subheader("Sample of 100 observations")
    st.write(df_sample)

# Define Features and Target Dataframes
 
  # Splittin Dataset into Categorical and numerical variables
  cat_data =[]
  num_data =[]
  for i,c in enumerate (df.dtypes):
    if c==object:
      cat_data.append(df.iloc[:,i])
    else:
      num_data.append(df.iloc[:,i])
  cat_data = pd.DataFrame(cat_data).transpose()
  num_data = pd.DataFrame(num_data).transpose()

  # Handle Missing Values
  cat_data = cat_data.apply(lambda x: x.fillna(x.value_counts().index[0]))
  num_data.fillna(method="bfill", inplace=True)

  # Target Encoding
  target_value = {"Y":1,"N":0}
  target = cat_data["Loan_Status"]
  cat_data.drop("Loan_Status",axis = 1,inplace = True)
  target = target.map(target_value)

  # Encoding others categorical variables
  Le = LabelEncoder()
  for i in cat_data:
    cat_data[i] = Le.fit_transform(cat_data[i])

  # Drop Loan_ID Column
  cat_data.drop("Loan_ID",axis = 1,inplace = True)

  # Define Features and Target Dataframes
  X = cat_data.join(num_data) 
  X = X.drop(["Married", "Dependents", "Loan_Amount_Term","Property_Area"],axis = 1)
  y = target
  
  
  st.sidebar.subheader("Features and Target Dataframes")
  if st.sidebar.checkbox("Features", False):
    st.subheader("Features Dataframe")
    st.write(X)
    st.write(X.shape)

  if st.sidebar.checkbox("Target", False):
    st.subheader("Target Dataframe")
    st.write(y) 
    st.write(y.shape) 
  
  # 4. Modeling
    #splitting data into training and test data
  seed = 42
  strat = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
  for train, test in strat.split(X, y):
    X_train, X_test = X.iloc[train], X.iloc[test]
    y_train, y_test = y.iloc[train], y.iloc[test]

  st.sidebar.subheader("Training and Test Data Shapes")
  if st.sidebar.checkbox("Training and Test Data Shapes", False):
    st.subheader("Training and Test Data Shapes")
    st.write("Shape X_train :", X_train.shape )
    st.write("Shape X_test :", X_test.shape)
    st.write("Shape y_train :", y_train.shape)
    st.write("Shape y_test :", y_test.shape)
  
  #Models Application
  models = {
    "Logistic Regression": LogisticRegression(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier()
  }
  st.sidebar.subheader("Models")
  if st.sidebar.checkbox("Models Application", False):
    for name, model in models.items():
      model.fit(X_train, y_train)
      y_pred = model.predict(X_test)
      st.subheader(f"{name} Model")
      st.write("Accuracy Score :", accuracy_score(y_test, y_pred))
  #6. Saving Models
  st.sidebar.subheader("Saving Models")
  if st.sidebar.button("Save Models", False):
    st.subheader("Saving Models")
    for name, model in models.items():
      joblib.dump(model, f"{name}.pkl")
      st.write(f"{name} Model Saved Successfully")


if __name__ == "__main__":
    main()


