import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="💼 AI Salary Predictor",
    layout="wide"
)

# -------------------------------
# Helper Functions & CSS
# -------------------------------
def exp_bucket(x):
    if x < 2:
        return 'Entry'
    elif x < 5:
        return 'Mid'
    elif x < 10:
        return 'Senior'
    else:
        return 'Expert'

st.markdown(
    """
    <style>
    .main-title { color: #1f77b4; font-size: 36px; font-weight: bold; text-align: center; }
    .prediction-card { 
        background-color: #ffffff !important; 
        border-radius: 10px; 
        padding: 20px; 
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1); 
        color: #1c1c1c !important; 
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# Load Data & Model (Cached)
# -------------------------------
@st.cache_resource
def get_trained_model():
    # 1. Load Data
    data = pd.read_csv("AI_Job_Market_Trends_2026.csv")
    if 'job_id' in data.columns:
        data = data.drop(columns=['job_id'])

    # 2. Feature Engineering
    skill_cols = ['skills_python', 'skills_sql', 'skills_ml', 'skills_deep_learning', 'skills_cloud']
    data['total_skills'] = data[skill_cols].sum(axis=1)
    data['experience_bucket'] = data['years_experience'].apply(exp_bucket)

    # 3. Model Training Setup
    TARGET = 'salary'
    X_train = data.drop(columns=[TARGET, 'job_posting_year', 'job_posting_month'], errors='ignore')
    y_train = data[TARGET]

    categorical_cols = X_train.select_dtypes(include=['object']).columns
    numerical_cols = X_train.select_dtypes(include=[np.number]).columns

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

    # 4. Pipeline
    trained_model = Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    trained_model.fit(X_train, y_train)
    
    return trained_model, data, numerical_cols, categorical_cols

# Initialize the model and data
model, df, numerical_cols, categorical_cols = get_trained_model()

# -------------------------------
# Streamlit Tabs
# -------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Inputs", "Prediction", "Skills Overview", "Feature Importance"])

# -------------------------------
# Inputs Tab
# -------------------------------
with tab1:
    st.markdown('<div class="main-title">💼 AI Salary Predictor</div>', unsafe_allow_html=True)
    st.header("Enter Job Details")

    col1, col2, col3 = st.columns(3)
    with col1:
        company_size = st.selectbox("Company Size", sorted(df['company_size'].unique()))
        company_industry = st.selectbox("Industry", sorted(df['company_industry'].unique()))
    with col2:
        country = st.selectbox("Country", sorted(df['country'].unique()))
        remote_type = st.selectbox("Remote Type", sorted(df['remote_type'].unique()))
    with col3:
        experience_level = st.selectbox("Experience Level", sorted(df['experience_level'].unique()))
        education_level = st.selectbox("Education Level", sorted(df['education_level'].unique()))

    years_experience = st.slider("Years of Experience", 0, 20, 2)

    col1, col2, col3, col4, col5 = st.columns(5)
    skills_python = col1.checkbox("Python")
    skills_sql = col2.checkbox("SQL")
    skills_ml = col3.checkbox("Machine Learning")
    skills_dl = col4.checkbox("Deep Learning")
    skills_cloud = col5.checkbox("Cloud")

    hiring_urgency = st.selectbox("Hiring Urgency", sorted(df['hiring_urgency'].unique()))
    job_openings = st.slider("Job Openings", 1, 20, 3)

# -------------------------------
# Prediction Tab
# -------------------------------
with tab2:
    st.header("Predicted Salary")

    skill_values = {
        'skills_python': int(skills_python),
        'skills_sql': int(skills_sql),
        'skills_ml': int(skills_ml),
        'skills_deep_learning': int(skills_dl),
        'skills_cloud': int(skills_cloud)
    }
    
    input_data = pd.DataFrame([{
        **skill_values,
        'job_title': 'Data Scientist',
        'company_size': company_size,
        'company_industry': company_industry,
        'country': country,
        'remote_type': remote_type,
        'experience_level': experience_level,
        'years_experience': years_experience,
        'education_level': education_level,
        'hiring_urgency': hiring_urgency,
        'job_openings': job_openings,
        'total_skills': sum(skill_values.values()),
        'experience_bucket': exp_bucket(years_experience)
    }])

    prediction = model.predict(input_data)[0]
    
    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
    st.write(f"**Experience Bucket:** {exp_bucket(years_experience)}")
    st.markdown(f"<h2 style='color:green;'>💰 Predicted Salary: ${prediction:,.0f}</h2>", unsafe_allow_html=True)
    st.markdown(f"**Estimated Salary Range (±10%):** ${prediction*0.9:,.0f} - ${prediction*1.1:,.0f}")
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# Skills Overview Tab
# -------------------------------
with tab3:
    st.header("Skills Overview")
    skill_labels = ['Python', 'SQL', 'Machine Learning', 'Deep Learning', 'Cloud']
    skill_numeric = [int(skills_python), int(skills_sql), int(skills_ml), int(skills_dl), int(skills_cloud)]
    skill_df = pd.DataFrame({'Skill': skill_labels, 'Status': ['Selected' if x else 'Not Selected' for x in skill_numeric]})
    st.table(skill_df)
    
    chart_data = pd.DataFrame({'Skill': skill_labels, 'Value': skill_numeric})
    st.bar_chart(chart_data.set_index('Skill'))

# -------------------------------
# Feature Importance Tab
# -------------------------------
with tab4:
    st.header("Top 10 Feature Importances")
    
    num_f = numerical_cols.tolist()
    cat_f = model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_cols)
    all_f = np.concatenate([num_f, cat_f])
    
    importances = model.named_steps['model'].feature_importances_
    feat_imp_df = pd.DataFrame({'Feature': all_f, 'Importance': importances}).sort_values('Importance', ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(8,5))
    ax.barh(feat_imp_df['Feature'], feat_imp_df['Importance'], color='#1f77b4')
    ax.invert_yaxis()
    st.pyplot(fig)
