# Libraries Importing
import streamlit as st
import pandas as pd


def main():
  st.title("DATA")
  st.title("LOAN BANK PREDICTION APP")
  st.sidebar.title("Loan Application Form")
# 1. Reading Data
  @st.cache_data
  def load_data():
    data= pd.read_csv("Loan_Bank.csv")
    return data
  df = load_data()
  df_sample = df.sample(100)
  if st.sidebar.checkbox("Show Dataset",False):
    st.subheader("Sample of 100 observations")
    st.write(df_sample)
    st.subheader("Datatset Descriptions")
    markdown_text = """
    - **Loan_ID**: A unique loan ID.
    - **Gender**: Either male or female.
    - **Married**: Weather Married (yes) or Not Married (No).
    - **Dependents**: Number of persons depending on the client.
    - **Education**: Applicant Education (Graduate or Undergraduate).
    - **Self_Employed**: Self-employed (Yes/No).
    - **ApplicantIncome**: Applicant income.
    - **CoapplicantIncome**: Co-applicant income.
    - **LoanAmount**: Loan amount in thousands.
    - **Loan_Amount_Term**: Terms of the loan in months.
    - **Credit_History**: Credit history meets guidelines.
    - **Property_Area**: Applicants are living either Urban, Semi-Urban or Rural.
    - **Loan_Status**: Loan approved (Y/N).
    """
    st.markdown(markdown_text)


if __name__ == "__main__":
    main()



