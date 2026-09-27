import os
import streamlit as st
import pandas as pd
import joblib



# Load the model committed by the pipeline (sits next to this file)
model_path = "tourism_project/deployment/Tourism_Package_Prediction.joblib"

model = joblib.load(model_path)

st.title("Tourism Package Prediction App")
st.write("""
This application predicts the likelihood of a tourism package 
""")


Age = st.number_input("Age")
TypeofContact = st.selectbox("TypeofContact",['Self Enquiry','Company Invited'])
CityTier = st.number_input("CityTier")
DurationOfPitch = st.number_input("DurationOfPitch")
Occupation = st.selectbox("Occupation",['Salaried','Free Lancer','Small Business','Large Business']) # Corrected syntax
Gender = st.selectbox("Gender",['Male','Female'])
NumberOfPersonVisiting = st.number_input("NumberOfPersonVisiting")
NumberOfFollowups = st.number_input("NumberOfFollowups")
ProductPitched = st.selectbox("ProductPitched",['Basic','Standard','Deluxe','King','Super Deluxe'])
PreferredPropertyStar = st.number_input("PreferredPropertyStar")
MaritalStatus = st.selectbox("MaritalStatus",['Married','Single','Divorced'])
NumberOfTrips = st.number_input("NumberOfTrips")
Passport = st.selectbox("Passport",['Yes','No'])
PitchSatisfactionScore = st.number_input("PitchSatisfactionScore")
OwnCar = st.number_input("OwnCar")
NumberOfChildrenVisiting = st.number_input("NumberOfChildrenVisiting")
Designation = st.selectbox("Designation",['Executive','Manager','Senior Manager','AVP','VP'])
MonthlyIncome = st.number_input("MonthlyIncome")


input_data = pd.DataFrame([{
   "Age" : Age,
   "TypeofContact" : TypeofContact,
   "CityTier" : CityTier,
   "DurationOfPitch" : DurationOfPitch,
   "Occupation" : Occupation,
   "Gender" : Gender,
   "NumberOfPersonVisiting" : NumberOfPersonVisiting,
   "NumberOfFollowups" : NumberOfFollowups,
   "ProductPitched" : ProductPitched,
   "PreferredPropertyStar" : PreferredPropertyStar,
   "MaritalStatus" : MaritalStatus,
   "NumberOfTrips" : NumberOfTrips,
   "Passport" : Passport,
   "PitchSatisfactionScore" : PitchSatisfactionScore,
   "OwnCar" : OwnCar,
   "NumberOfChildrenVisiting" : NumberOfChildrenVisiting,
   "Designation" : Designation,
   "MonthlyIncome" : MonthlyIncome
}])

if st.button("Predict Purchase"): # Changed button text
    prediction = model.predict(input_data)[0]
    result = "Tourism Package Purchased" if prediction == 1 else "No Package Purchased" # Corrected messages
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
