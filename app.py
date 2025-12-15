import streamlit as st
import pandas as pd
import pickle
from sklearn.pipeline import Pipeline

# Load the trained model
@st.cache_resource
def load_model():
    with open('MLR_USING_PIPE.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file 'MLR_USING_PIPE.pkl' not found. Please ensure it is in the same directory.")
    st.stop()

# Title and description
st.title("Salary Prediction App")
st.write("Enter employee details to predict the estimated salary.")

# Input form
with st.form("prediction_form"):
    st.header("Employee Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        years_experience = st.number_input("Years of Experience", min_value=0.0, max_value=50.0, value=1.0, step=0.1)
        age = st.number_input("Age", min_value=18, max_value=100, value=25, step=1)
        education = st.selectbox("Education Level", ['Bachelor', 'Master', 'PhD'])
        job_type = st.selectbox("Job Type", ['HR', 'Sales', 'Marketing', 'Engineering', 'DataScience', 'Management'])
        
    with col2:
        location = st.selectbox("Location", ['Rural', 'Suburban', 'Urban'])
        company_size = st.selectbox("Company Size", ['Small', 'Medium', 'Large'])
        performance_rating = st.slider("Performance Rating", min_value=1, max_value=5, value=3)
        projects_completed = st.number_input("Projects Completed", min_value=0, max_value=100, value=5, step=1)
        
    col3, col4 = st.columns(2)
    with col3:
        certifications = st.number_input("Certifications", min_value=0, max_value=20, value=0, step=1)
    with col4:
        promoted = st.selectbox("Promoted Recently?", ['No', 'Yes'])

    submitted = st.form_submit_button("Predict Salary")

# Prediction logic
if submitted:
    # Create a DataFrame from inputs
    input_data = pd.DataFrame({
        'YearsExperience': [years_experience],
        'Age': [age],
        'Education': [education],
        'JobType': [job_type],
        'Location': [location],
        'CompanySize': [company_size],
        'PerformanceRating': [performance_rating],
        'ProjectsCompleted': [projects_completed],
        'Certifications': [certifications],
        'Promoted': [promoted]
    })

    # Make prediction
    try:
        prediction = model.predict(input_data)
        salary = prediction[0]
        st.success(f"Estimated Salary: ${salary:,.2f}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
