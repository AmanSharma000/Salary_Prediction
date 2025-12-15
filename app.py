import streamlit as st
import pandas as pd
import pickle

# Page Configuration
st.set_page_config(
    page_title="Salary Prediction Pro",
    page_icon="💼",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #2e86de;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        height: 48px;
        font-size: 16px;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #0984e3;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .input-card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    .metric-container {
        background: linear-gradient(135deg, #6c5ce7 0%, #a29bfe 100%);
        padding: 30px;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 20px rgba(108, 92, 231, 0.2);
    }
    h1, h2, h3 {
        color: #2d3436;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_model():
    with open('MLR_USING_PIPE.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

try:
    model = load_model()
except FileNotFoundError:
    st.error("⚠️ Model file 'MLR_USING_PIPE.pkl' not found. Please ensure it is in the same directory.")
    st.stop()

# Header
st.markdown("<h1 style='text-align: center; margin-bottom: 30px;'>💼 Salary Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #636e72; font-size: 18px; margin-bottom: 40px;'>Enter employee details below to estimate annual compensation.</p>", unsafe_allow_html=True)

# Main Form
with st.form("prediction_form"):
    # Section 1: Personal & Education
    st.markdown("### 👤 Personal & Education")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age", 18, 65, 30)
    with c2:
        education = st.selectbox("Education Level", ['Bachelor', 'Master', 'PhD'])
    with c3:
        location = st.selectbox("Location", ['Rural', 'Suburban', 'Urban'])
    
    st.markdown("---")
    
    # Section 2: Professional Experience
    st.markdown("### 🏢 Professional Experience")
    c4, c5, c6 = st.columns(3)
    with c4:
        job_type = st.selectbox("Job Role", ['HR', 'Sales', 'Marketing', 'Engineering', 'DataScience', 'Management'])
    with c5:
        years_experience = st.number_input("Years of Experience", 0.0, 50.0, 5.0, 0.5)
    with c6:
        company_size = st.select_slider("Company Size", options=['Small', 'Medium', 'Large'])

    st.markdown("---")

    # Section 3: Performance & Achievements
    st.markdown("### 📈 Performance & Metrics")
    c7, c8, c9, c10 = st.columns(4)
    with c7:
        performance_rating = st.slider("Performance (1-5)", 1, 5, 3)
    with c8:
        projects_completed = st.number_input("Projects Done", 0, 100, 10)
    with c9:
        certifications = st.number_input("Certifications", 0, 20, 1)
    with c10:
        promoted = st.selectbox("Recently Promoted?", ['No', 'Yes'])

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Submit Button
    submitted = st.form_submit_button("Generate Prediction 🚀")

# Prediction Logic
if submitted:
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

    try:
        prediction = model.predict(input_data)[0]
        
        st.markdown("---")
        
        # Result Display
        col_spacer_l, col_res, col_spacer_r = st.columns([1, 2, 1])
        with col_res:
            st.markdown(f"""
                <div class="metric-container">
                    <h3 style="color: rgba(255,255,255,0.9); margin: 0;">Estimated Annual Salary</h3>
                    <h1 style="color: white; font-size: 56px; margin: 10px 0;">${prediction:,.2f}</h1>
                    <p style="color: rgba(255,255,255,0.8); margin: 0;">Based on {years_experience} years of experience in {job_type}</p>
                </div>
            """, unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"An error occurred: {e}")
