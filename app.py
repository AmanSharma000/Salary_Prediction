import streamlit as st
import pandas as pd
import pickle

# Page Configuration
st.set_page_config(
    page_title="Salary Prediction Pro",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        height: 50px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    h1 {
        color: #2c3e50;
        font-family: 'Helvetica Neue', sans-serif;
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

# Sidebar Inputs
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100)
    st.title("Employee Profile")
    st.markdown("---")
    
    st.subheader("Personal Details")
    age = st.slider("Age", 18, 65, 30)
    education = st.selectbox("Education Level", ['Bachelor', 'Master', 'PhD'])
    location = st.selectbox("Location", ['Rural', 'Suburban', 'Urban'])
    
    st.subheader("Professional Details")
    job_type = st.selectbox("Job Role", ['HR', 'Sales', 'Marketing', 'Engineering', 'DataScience', 'Management'])
    years_experience = st.number_input("Years of Experience", 0.0, 50.0, 5.0, 0.5)
    company_size = st.select_slider("Company Size", options=['Small', 'Medium', 'Large'])
    
    st.subheader("Performance Metrics")
    performance_rating = st.slider("Performance Rating (1-5)", 1, 5, 3)
    projects_completed = st.number_input("Projects Completed", 0, 100, 10)
    certifications = st.number_input("Certifications", 0, 20, 1)
    promoted = st.radio("Recently Promoted?", ['No', 'Yes'], horizontal=True)
    
    st.markdown("---")
    predict_btn = st.button("Predict Salary 🚀")

# Main Content
st.title("💼 Salary Prediction System")
st.markdown("### Industry-Grade Machine Learning Pipeline")
st.markdown("Enter the employee details in the sidebar to generate a salary estimate based on our advanced ML model.")

# Prediction Logic
if predict_btn:
    # Create DataFrame
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
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown(f"""
                <div class="metric-card">
                    <h3 style="margin-bottom: 0px; color: #7f8c8d;">Estimated Annual Salary</h3>
                    <h1 style="color: #27ae60; font-size: 48px; margin-top: 10px;">${prediction:,.2f}</h1>
                    <p style="color: #95a5a6;">Based on current market trends</p>
                </div>
            """, unsafe_allow_html=True)
            
        st.success("✅ Prediction generated successfully!")
        
        # Optional: Feature breakdown (visual flair)
        with st.expander("See Input Summary"):
            st.json(input_data.to_dict(orient='records')[0])
            
    except Exception as e:
        st.error(f"An error occurred: {e}")

else:
    st.info("👈 Please configure the employee profile in the sidebar and click 'Predict Salary'.")
