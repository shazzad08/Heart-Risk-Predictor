import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load saved model, scaler, and expected columns
@st.cache_resource
def load_assets():
    model = joblib.load("KNN_heart.pkl")
    scaler = joblib.load("scaler.pkl")
    expected_columns = joblib.load("columns.pkl")
    return model, scaler, expected_columns

model, scaler, expected_columns = load_assets()

# Custom CSS for UI styling
st.markdown("""
<style>
/* Main background and text */
.stApp {
    background-color: #f4f6f9;
}

/* Hide standard menu and footer, but keep header for sidebar toggle */
#MainMenu {visibility: hidden;}
[data-testid="stHeader"] {background-color: transparent;}
.stDeployButton {display:none;}
[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e0e0e0;
}
[data-testid="stSidebarHeader"] {
    padding: 0;
}
.sidebar-header {
    background-color: #c60000;
    color: white;
    padding: 20px;
    text-align: left;
    display: flex;
    align-items: center;
    gap: 15px;
    margin-top: -60px;
    margin-left: -1rem;
    margin-right: -1rem;
    margin-bottom: 20px;
}
.sidebar-header h2 {
    color: white;
    margin: 0;
    font-size: 1.6rem;
    line-height: 1.2;
    font-weight: 700;
}

/* Card styling */
.custom-card {
    background-color: white;
    border-radius: 10px;
    padding: 25px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    margin-bottom: 20px;
    border: 1px solid #f0f0f0;
    height: 100%;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.custom-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 25px rgba(0,0,0,0.1);
}
.prediction-card-high {
    background-color: #fff1f0;
    border: 1px solid #ffccc7;
    border-radius: 10px;
    padding: 30px;
    text-align: center;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 20px;
}
.prediction-card-low {
    background-color: #f6ffed;
    border: 1px solid #b7eb8f;
    border-radius: 10px;
    padding: 30px;
    text-align: center;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 20px;
}
.info-box {
    background-color: #f0f7ff;
    border: 1px solid #bae0ff;
    border-radius: 8px;
    padding: 12px 15px;
    color: #0050b3;
    font-size: 0.9rem;
    margin-top: 20px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.gradient-bar-container {
    width: 100%;
    height: 15px;
    background: linear-gradient(to right, #52c41a, #faad14, #f5222d);
    border-radius: 10px;
    position: relative;
    margin-top: 25px;
    margin-bottom: 5px;
}
.gradient-bar-marker {
    width: 0; 
    height: 0; 
    border-left: 8px solid transparent;
    border-right: 8px solid transparent;
    border-top: 12px solid #262626;
    position: absolute;
    top: -12px;
    transform: translateX(-50%);
}
.gradient-labels {
    display: flex;
    justify-content: space-between;
    font-size: 0.8rem;
    color: #8c8c8c;
    margin-bottom: 15px;
}
.risk-alert {
    background-color: #fff1f0;
    color: #cf1322;
    padding: 12px;
    border-radius: 8px;
    text-align: center;
    font-weight: 500;
    font-size: 0.9rem;
    margin-top: 15px;
}
.risk-alert-low {
    background-color: #f6ffed;
    color: #389e0d;
    padding: 12px;
    border-radius: 8px;
    text-align: center;
    font-weight: 500;
    font-size: 0.9rem;
    margin-top: 15px;
}
.health-overview-card {
    background-color: white;
    border-radius: 10px;
    padding: 20px 15px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    border: 1px solid #f0f0f0;
    display: flex;
    align-items: center;
    gap: 15px;
    height: 100%;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.health-overview-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 20px rgba(0,0,0,0.08);
}
.health-overview-icon {
    font-size: 1.8rem;
    background-color: #f8f9fa;
    width: 50px;
    height: 50px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
}
.health-overview-text h4 {
    margin: 0 0 5px 0;
    font-size: 0.8rem;
    color: #595959;
    font-weight: 600;
}
.health-overview-text h2 {
    margin: 0;
    font-size: 1.6rem;
    color: #262626;
    line-height: 1;
    font-weight: 700;
}
.health-overview-text p {
    margin: 0;
    font-size: 0.75rem;
    color: #8c8c8c;
}
.recommendation-item {
    display: flex;
    gap: 15px;
    margin-bottom: 15px;
    padding: 15px;
    border-radius: 8px;
    transition: transform 0.3s ease, background-color 0.3s ease;
}
.recommendation-item.high-risk {
    background-color: #fff5f5;
}
.recommendation-item.high-risk:hover {
    transform: translateX(8px);
    background-color: #ffe8e8;
}
.recommendation-item.low-risk {
    background-color: #f0f7ff;
}
.recommendation-item.low-risk:hover {
    transform: translateX(8px);
    background-color: #e6f4ff;
}
.recommendation-item-icon {
    font-size: 1.6rem;
}
.recommendation-item-text h4 {
    margin: 0 0 4px 0;
    font-size: 0.95rem;
    color: #262626;
    font-weight: 600;
}
.recommendation-item-text p {
    margin: 0;
    font-size: 0.85rem;
    color: #595959;
}
.tips-item {
    display: flex;
    gap: 12px;
    margin-bottom: 20px;
    padding: 10px;
    border-radius: 8px;
    margin-left: -10px;
    margin-right: -10px;
    transition: transform 0.3s ease, background-color 0.3s ease;
}
.tips-item.high-risk:hover {
    transform: translateX(8px);
    background-color: #fff1f0;
}
.tips-item.low-risk:hover {
    transform: translateX(8px);
    background-color: #f6ffed;
}
.tips-item-icon {
    font-size: 1.3rem;
}
.tips-item-text h4 {
    margin: 0 0 4px 0;
    font-size: 0.95rem;
    color: #262626;
    font-weight: 600;
}
.tips-item-text p {
    margin: 0;
    font-size: 0.85rem;
    color: #595959;
}
/* Style the main title */
.main-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 30px;
    margin-top: 20px;
}
.main-title {
    display: flex;
    align-items: center;
    gap: 15px;
}
.main-title h1 {
    margin: 0;
    color: #1a202c;
    font-size: 2.2rem;
    font-weight: 800;
}
.main-title p {
    margin: 5px 0 0 0;
    color: #64748b;
    font-size: 1.1rem;
}
.secure-badge {
    background-color: #fff1f0;
    color: #c60000;
    padding: 8px 18px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.9rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.footer {
    text-align: center;
    padding: 30px 20px;
    color: #8c8c8c;
    font-size: 0.95rem;
    margin-top: 50px;
    border-top: 1px solid #e8e8e8;
}
/* Button styling */
div.stButton > button:first-child {
    background-color: #c60000;
    color: white;
    width: 100%;
    border-radius: 8px;
    padding: 12px;
    font-weight: bold;
    font-size: 1.05rem;
    border: none;
    transition: all 0.3s;
}
div.stButton > button:first-child:hover {
    background-color: #a00000;
    color: white;
    box-shadow: 0 4px 8px rgba(198, 0, 0, 0.2);
}
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR -----------------
st.sidebar.markdown("""
<div class="sidebar-header">
    <div style="font-size: 2.5rem;">🤍</div>
    <div>
        <h2 style="margin:0;">Heart Risk</h2>
        <div style="font-size:1rem; opacity:0.9;">Predictor</div>
    </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar.expander("👤 Patient Information", expanded=True):
    age = st.slider("📅 Age", 18, 100, 40)
    sex = st.selectbox("⚧ Sex", ["M", "F"])
    chest_pain = st.selectbox("❤️ Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
    resting_bp = st.number_input("🩺 Resting Blood Pressure (mm Hg)", 80, 200, 120)
    cholesterol = st.number_input("💧 Cholesterol (mg/dL)", 100, 600, 200)
    fasting_bs = st.selectbox("🩸 Fasting Blood Sugar > 120 mg/dL", ["0 (False)", "1 (True)"])
    resting_ecg = st.selectbox("📉 Resting ECG", ["Normal", "ST", "LVH"])
    max_hr = st.slider("💓 Max Heart Rate", 60, 220, 150)
    exercise_angina = st.selectbox("🏃 Exercise-Induced Angina", ["N", "Y"])
    oldpeak = st.slider("📈 Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
    st_slope = st.selectbox("📉 ST Slope", ["Up", "Flat", "Down"])

st.sidebar.markdown("<hr style='margin: 10px 0 20px 0;'>", unsafe_allow_html=True)
predict_button = st.sidebar.button("🩺 Predict Risk", use_container_width=True)

# ----------------- MAIN AREA -----------------
st.markdown("""
<div class="main-header">
    <div class="main-title">
        <div style="font-size: 2.8rem; color: #c60000;">❤️</div>
        <div>
            <h1>Heart Disease Risk Predictor</h1>
            <p>AI-powered early heart disease risk screening system</p>
        </div>
    </div>
    <div class="secure-badge">
        <span style="color: #c60000;">🛡️</span> Secure • Reliable • Accurate
    </div>
</div>
""", unsafe_allow_html=True)

if predict_button:
    fasting_bs_val = 0 if "0" in fasting_bs else 1
    
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs_val,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected_columns]
    scaled_input = scaler.transform(input_df)
    
    prediction = model.predict(scaled_input)[0]
    probabilities = model.predict_proba(scaled_input)[0]
    
    st.session_state.prediction_made = True
    st.session_state.is_high_risk = bool(prediction == 1)
    st.session_state.risk_prob = float(probabilities[1]) * 100

# Top Cards
col1, col2 = st.columns(2)

with col1:
    st.markdown("<h4 style='color: #001529; font-size: 1.15rem; margin-bottom: 15px;'>Your Risk Prediction</h4>", unsafe_allow_html=True)
    if not st.session_state.get('prediction_made', False):
        st.markdown(
"""<div class="custom-card" style="text-align: center; color: #8c8c8c; display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 250px;">
    <div style="font-size: 3.5rem; margin-bottom: 15px; opacity: 0.5;">📊</div>
    <h3 style="margin: 0; font-weight: 500;">Awaiting Input</h3>
    <p style="margin-top: 5px;">Fill the patient information and click Predict Risk.</p>
</div>""", unsafe_allow_html=True)
    elif st.session_state.is_high_risk:
        st.markdown(
"""<div class="custom-card" style="display: flex; flex-direction: column; justify-content: center; min-height: 250px; padding: 20px;">
    <div class="prediction-card-high">
        <div style="font-size: 4.5rem; color: #cf1322; line-height: 1;">⚠️</div>
        <div style="text-align: left;">
            <h2 style="margin: 0; color: #cf1322; font-size: 2.2rem; font-weight: 800;">High Risk</h2>
            <h3 style="margin: 0; color: #001529; font-weight: 500; font-size: 1.3rem;">of Heart Disease</h3>
        </div>
    </div>
    <div class="info-box">
        <span style="font-size: 1.2rem;">ℹ️</span> This prediction is based on the provided health information and machine learning analysis.
    </div>
</div>""", unsafe_allow_html=True)
    else:
        st.markdown(
"""<div class="custom-card" style="display: flex; flex-direction: column; justify-content: center; min-height: 250px; padding: 20px;">
    <div class="prediction-card-low">
        <div style="font-size: 4.5rem; color: #389e0d; line-height: 1;">✅</div>
        <div style="text-align: left;">
            <h2 style="margin: 0; color: #389e0d; font-size: 2.2rem; font-weight: 800;">Low Risk</h2>
            <h3 style="margin: 0; color: #001529; font-weight: 500; font-size: 1.3rem;">of Heart Disease</h3>
        </div>
    </div>
    <div class="info-box">
        <span style="font-size: 1.2rem;">ℹ️</span> This prediction is based on the provided health information and machine learning analysis.
    </div>
</div>""", unsafe_allow_html=True)

with col2:
    st.markdown("<h4 style='color: #001529; font-size: 1.15rem; margin-bottom: 15px;'>Risk Probability</h4>", unsafe_allow_html=True)
    if not st.session_state.get('prediction_made', False):
        st.markdown(
"""<div class="custom-card" style="text-align: center; color: #8c8c8c; display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 250px;">
    <div style="font-size: 3.5rem; margin-bottom: 15px; opacity: 0.5;">⏳</div>
    <h3 style="margin: 0; font-weight: 500;">Awaiting Input</h3>
    <p style="margin-top: 5px;">Waiting for prediction...</p>
</div>""", unsafe_allow_html=True)
    else:
        is_high = st.session_state.is_high_risk
        prob = st.session_state.risk_prob
        prob_color = "#cf1322" if is_high else "#389e0d"
        prob_text = "High Risk" if is_high else "Low Risk"
        alert_class = "risk-alert" if is_high else "risk-alert-low"
        
        if is_high:
            alert_text = f"Your risk is higher than {prob:.0f}% of similar cases"
            icon = "📈"
        else:
            alert_text = f"Your risk is lower than {100-prob:.0f}% of similar cases"
            icon = "📉"
            
        st.markdown(
f"""<div class="custom-card" style="text-align: center; display: flex; flex-direction: column; justify-content: center; min-height: 250px; padding: 30px;">
<h1 style="color: {prob_color}; font-size: 4rem; margin: 0; font-weight: 800; line-height: 1;">{prob:.1f}%</h1>
<h3 style="color: {prob_color}; margin: 5px 0 0 0; font-size: 1.3rem; font-weight: 600;">{prob_text}</h3>
<div class="gradient-bar-container">
<div class="gradient-bar-marker" style="left: {prob}%;"></div>
</div>
<div class="gradient-labels">
<span>0%</span>
<span>50%</span>
<span>100%</span>
</div>
<div class="{alert_class}">
{icon} {alert_text}
</div>
</div>""", unsafe_allow_html=True)

if st.session_state.get('prediction_made', False):
    # Health Overview
    st.markdown("<h4 style='color: #001529; font-size: 1.15rem; margin-top: 25px; margin-bottom: 15px;'><span style='color:#c60000; margin-right: 8px;'>📈</span> Health Overview</h4>", unsafe_allow_html=True)
    
    h_col1, h_col2, h_col3, h_col4, h_col5, h_col6 = st.columns(6)
    
    def overview_card(icon, title, value, unit, icon_bg, icon_color):
        return f"""
        <div class="health-overview-card">
            <div class="health-overview-icon" style="background-color: {icon_bg}; color: {icon_color};">
                {icon}
            </div>
            <div class="health-overview-text">
                <h4>{title}</h4>
                <h2>{value}</h2>
                <p>{unit}</p>
            </div>
        </div>
        """
    
    with h_col1:
        st.markdown(overview_card("👤", "Age", age, "years", "#f0f5ff", "#2f54eb"), unsafe_allow_html=True)
    with h_col2:
        st.markdown(overview_card("🫀", "Blood Pressure", resting_bp, "mm Hg", "#f6ffed", "#52c41a"), unsafe_allow_html=True)
    with h_col3:
        st.markdown(overview_card("💧", "Cholesterol", cholesterol, "mg/dL", "#fffbe6", "#faad14"), unsafe_allow_html=True)
    with h_col4:
        st.markdown(overview_card("❤️", "Max Heart Rate", max_hr, "bpm", "#fff1f0", "#f5222d"), unsafe_allow_html=True)
    with h_col5:
        st.markdown(overview_card("📈", "Oldpeak", oldpeak, "", "#e6f7ff", "#1890ff"), unsafe_allow_html=True)
    with h_col6:
        st.markdown(overview_card("📉", "ST Slope", st_slope, "", "#f9f0ff", "#722ed1"), unsafe_allow_html=True)
    
    # Recommendations & Tips
    r_col1, r_col2 = st.columns(2)
    
    with r_col1:
        st.markdown("<h4 style='color: #001529; font-size: 1.15rem; margin-top: 25px; margin-bottom: 15px;'><span style='color:#c60000; margin-right: 8px;'>🛡️</span> Recommendations</h4>", unsafe_allow_html=True)
        if st.session_state.is_high_risk:
            st.markdown(
"""<div class="custom-card">
    <div class="recommendation-item high-risk">
        <div class="recommendation-item-icon">👨‍⚕️</div>
        <div class="recommendation-item-text">
            <h4>Consult a cardiologist immediately</h4>
            <p>Professional medical evaluation is highly recommended</p>
        </div>
    </div>
    <div class="recommendation-item high-risk">
        <div class="recommendation-item-icon">🧪</div>
        <div class="recommendation-item-text">
            <h4>Complete diagnostic tests</h4>
            <p>ECG, Stress Test, and Angiography if recommended</p>
        </div>
    </div>
    <div class="recommendation-item high-risk">
        <div class="recommendation-item-icon">💊</div>
        <div class="recommendation-item-text">
            <h4>Strictly follow prescribed medications</h4>
            <p>Do not skip any doses of your current medication</p>
        </div>
    </div>
</div>""", unsafe_allow_html=True)
        else:
            st.markdown(
"""<div class="custom-card">
    <div class="recommendation-item low-risk">
        <div class="recommendation-item-icon">👨‍⚕️</div>
        <div class="recommendation-item-text">
            <h4>Maintain regular health checkups</h4>
            <p>An annual physical exam is sufficient</p>
        </div>
    </div>
    <div class="recommendation-item low-risk">
        <div class="recommendation-item-icon">📊</div>
        <div class="recommendation-item-text">
            <h4>Monitor vitals periodically</h4>
            <p>Keep your blood pressure and cholesterol in check</p>
        </div>
    </div>
    <div class="recommendation-item low-risk">
        <div class="recommendation-item-icon">🏃‍♂️</div>
        <div class="recommendation-item-text">
            <h4>Stay active and aware</h4>
            <p>No immediate medical intervention needed</p>
        </div>
    </div>
</div>""", unsafe_allow_html=True)
    
    with r_col2:
        st.markdown("<h4 style='color: #001529; font-size: 1.15rem; margin-top: 25px; margin-bottom: 15px;'><span style='color:#52c41a; margin-right: 8px;'>💚</span> Heart Health Tips</h4>", unsafe_allow_html=True)
        if st.session_state.is_high_risk:
            st.markdown(
"""<div class="custom-card" style="padding: 30px;">
    <div class="tips-item high-risk">
        <div class="tips-item-icon" style="color: #cf1322;">🫀</div>
        <div class="tips-item-text">
            <h4>Supervised Exercise</h4>
            <p>Engage in cardiac rehab or light activities</p>
        </div>
    </div>
    <div class="tips-item high-risk">
        <div class="tips-item-icon" style="color: #cf1322;">🥗</div>
        <div class="tips-item-text">
            <h4>Strict Heart-Healthy Diet</h4>
            <p>Zero trans-fats, very low sodium, plant-based foods</p>
        </div>
    </div>
    <div class="tips-item high-risk">
        <div class="tips-item-icon" style="color: #cf1322;">🚫</div>
        <div class="tips-item-text">
            <h4>Immediate Lifestyle Changes</h4>
            <p>Quit smoking entirely and avoid all alcohol</p>
        </div>
    </div>
    <div class="tips-item high-risk">
        <div class="tips-item-icon" style="color: #cf1322;">📈</div>
        <div class="tips-item-text">
            <h4>Monitor Vitals Daily</h4>
            <p>Keep a daily log of blood pressure and heart rate</p>
        </div>
    </div>
</div>""", unsafe_allow_html=True)
        else:
            st.markdown(
"""<div class="custom-card" style="padding: 30px;">
    <div class="tips-item low-risk">
        <div class="tips-item-icon" style="color: #52c41a;">🏃‍♂️</div>
        <div class="tips-item-text">
            <h4>Exercise regularly</h4>
            <p>At least 30 minutes of moderate exercise daily</p>
        </div>
    </div>
    <div class="tips-item low-risk">
        <div class="tips-item-icon" style="color: #52c41a;">🥗</div>
        <div class="tips-item-text">
            <h4>Eat a balanced diet</h4>
            <p>Include whole grains, lean proteins, and veggies</p>
        </div>
    </div>
    <div class="tips-item low-risk">
        <div class="tips-item-icon" style="color: #cf1322;">🚫</div>
        <div class="tips-item-text">
            <h4>Avoid smoking and limit alcohol</h4>
            <p>Preventive measures for long-term health</p>
        </div>
    </div>
    <div class="tips-item low-risk">
        <div class="tips-item-icon" style="color: #52c41a;">🧘‍♂️</div>
        <div class="tips-item-text">
            <h4>Manage stress</h4>
            <p>Practice meditation and maintain work-life balance</p>
        </div>
    </div>
</div>""", unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <span style="color: #c60000;">❤️</span> Built with Streamlit & Machine Learning &nbsp;|&nbsp; Made by <span style="color: #c60000; font-weight: bold;">Shazzad</span>
</div>
""", unsafe_allow_html=True)