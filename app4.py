import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700&family=DM+Sans:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'Sora', sans-serif !important;
}

/* ── Streamlit default header is KEPT (no hiding rules) ── */
/* Only hide the deploy button to keep things clean */
.stDeployButton {
    display: none !important;
}

/* Background — dark charcoal with warm green undertones */
.stApp {
    background: linear-gradient(135deg, #1a1c18, #22251e, #1c2420);
    color: #e8eaf0;
}

/* ── Scrollbar ── */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-track {
    background: rgba(255,255,255,0.06);
    border-radius: 8px;
}
::-webkit-scrollbar-thumb {
    background: rgba(255,255,255,0.6);
    border-radius: 8px;
    border: 2px solid transparent;
    background-clip: content-box;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(255,255,255,0.9);
    background-clip: content-box;
}

/* Streamlit header — visible on dark background */
header[data-testid="stHeader"] {
    background: rgba(26, 28, 24, 0.95) !important;
    backdrop-filter: blur(8px);
    border-bottom: 1px solid rgba(255,255,255,0.1);
    height: auto !important;
    visibility: visible !important;
}
header[data-testid="stHeader"] * {
    color: #1a1c18 !important;
    fill: #1a1c18 !important;
    visibility: visible !important;
}
/* Keep header background light so text is readable */
header[data-testid="stHeader"] {
    background: rgba(232, 234, 240, 0.97) !important;
    border-bottom: 1px solid rgba(0,0,0,0.1) !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.05);
    border-right: 1px solid rgba(255,255,255,0.1);
    backdrop-filter: blur(10px);
}

section[data-testid="stSidebar"] * {
    color: #e8eaf0 !important;
}

/* ── Sliders — blue instead of default red ── */
[data-testid="stSlider"] div[data-baseweb="slider"] div[role="slider"] {
    background-color: #38bdf8 !important;
    border-color: #38bdf8 !important;
}
/* Filled track */
[data-testid="stSlider"] div[data-baseweb="slider"] div:nth-child(3) div {
    background: #38bdf8 !important;
}
/* Track thumb */
div[data-baseweb="slider"] [role="slider"] {
    background: #38bdf8 !important;
    border: 3px solid #38bdf8 !important;
    box-shadow: 0 0 0 4px rgba(56,189,248,0.25) !important;
}
/* Active filled portion of track */
div[data-baseweb="slider"] div[class*="Track"] > div:first-child {
    background: linear-gradient(90deg, #38bdf8, #6366f1) !important;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 16px;
    backdrop-filter: blur(8px);
}

/* Predict button — teal-to-indigo gradient (subtle, professional) */
div.stButton > button {
    background: linear-gradient(90deg, #38bdf8, #6366f1);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 14px 28px;
    font-size: 16px;
    font-weight: 600;
    font-family: 'Sora', sans-serif;
    width: 100%;
    transition: all 0.3s ease;
    box-shadow: 0 4px 20px rgba(99,102,241,0.35);
}
div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(99,102,241,0.55);
}

/* Score card */
.score-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 20px;
    padding: 32px;
    text-align: center;
    backdrop-filter: blur(12px);
    margin: 20px 0;
}

.score-number {
    font-family: 'Sora', sans-serif;
    font-size: 72px;
    font-weight: 700;
    background: linear-gradient(90deg, #38bdf8, #6366f1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1;
}

.score-label {
    font-size: 18px;
    color: rgba(255,255,255,0.55);
    margin-top: 8px;
}

/* Tips */
.tip-card {
    background: rgba(255,255,255,0.05);
    border-left: 4px solid #38bdf8;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 14px;
    color: rgba(255,255,255,0.85);
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}

[data-testid="stDataFrame"] table {
    color: white !important;
}

[data-testid="stDataFrame"] th,
[data-testid="stDataFrame"] thead tr th,
[data-testid="stDataFrame"] [role="columnheader"],
[data-testid="stDataFrame"] span.column-header-string {
    color: #ffffff !important;
    background-color: rgba(99,102,241,0.45) !important;
    font-family: 'Sora', sans-serif !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    opacity: 1 !important;
}

[data-testid="stDataFrame"] td,
[data-testid="stDataFrame"] [role="gridcell"] {
    color: #ffffff !important;
    background-color: rgba(255,255,255,0.04) !important;
    opacity: 1 !important;
}
[data-testid="stDataFrame"] tr:hover td {
    background-color: rgba(99,102,241,0.15) !important;
}

/* Metric text */
[data-testid="metric-container"] label,
[data-testid="stMetricLabel"] p,
[data-testid="stMetricValue"] {
    color: white !important;
    opacity: 1 !important;
}

/* Number input */
input[type="number"] {
    color: black !important;
    background-color: white !important;
}

/* Selectbox */
[data-testid="stSelectbox"] div[data-baseweb="select"] div,
[data-testid="stSelectbox"] span,
div[data-baseweb="select"] * {
    color: white !important;
}

ul[data-testid="stSelectboxVirtualDropdown"] li,
[role="option"] {
    color: white !important;
    background-color: #22251e !important;
}

div[data-baseweb="select"] > div {
    border-color: rgba(255,255,255,0.25) !important;
    background-color: rgba(255,255,255,0.07) !important;
}

/* Divider */
hr {
    border-color: rgba(255,255,255,0.1) !important;
}

/* Welcome card */
.welcome-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 20px;
    padding: 48px;
    text-align: center;
    margin-top: 60px;
}

.welcome-card h2 {
    font-size: 28px;
    color: rgba(255,255,255,0.9);
    margin-bottom: 12px;
}

.welcome-card p {
    color: rgba(255,255,255,0.5);
    font-size: 16px;
}

/* Footer */
.footer {
    text-align: center;
    padding: 24px;
    color: rgba(255,255,255,0.35);
    font-size: 13px;
    margin-top: 40px;
}

.footer a {
    color: #38bdf8;
    text-decoration: none;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Load model (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("student_lr_final_model.pkl", "rb") as file:
        model, scaler, le = pickle.load(file)
    return model, scaler, le


# ─────────────────────────────────────────────
# Preprocess
# ─────────────────────────────────────────────
def preprocessing_input_data(data, scaler, le):
    data['Extracurricular Activities'] = le.transform(
        [data['Extracurricular Activities']]
    )[0]

    df = pd.DataFrame([data])

    col_order = [
        'Hours Studied',
        'Previous Scores',
        'Extracurricular Activities',
        'Sleep Hours',
        'Sample Question Papers Practiced'
    ]
    df = df[col_order]

    df_transformed = scaler.transform(df)
    return df_transformed


# ─────────────────────────────────────────────
# Predict
# ─────────────────────────────────────────────
def predict_data(data):
    model, scaler, le = load_model()
    processed_data = preprocessing_input_data(data, scaler, le)
    prediction = model.predict(processed_data)
    return prediction


# ─────────────────────────────────────────────
# Grade helper
# ─────────────────────────────────────────────
def get_grade(score):
    if score >= 85:
        return "A+", "🏆", "#22c55e"
    elif score >= 75:
        return "A", "🎯", "#84cc16"
    elif score >= 60:
        return "B", "📈", "#eab308"
    elif score >= 50:
        return "C", "💪", "#f97316"
    else:
        return "D", "📚", "#ef4444"


# ─────────────────────────────────────────────
# Tips helper
# ─────────────────────────────────────────────
def get_tips(hour_studied, sleeping_hour, number_of_paper_solved, extra):
    tips = []
    if hour_studied < 5:
        tips.append("📚 Try studying at least 6 hours daily for better retention.")
    if sleeping_hour < 7:
        tips.append("😴 Aim for 7–8 hours of sleep — it boosts memory and focus!")
    if number_of_paper_solved < 3:
        tips.append("📄 Practice more question papers to build exam confidence.")
    if extra == "No":
        tips.append("🏆 Joining extracurricular activities helps with time management and overall growth.")
    if not tips:
        tips.append("🌟 Great habits! Keep maintaining your current routine.")
    return tips


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():

    # ── Header ──
    st.markdown("""
        <h1 style='font-family:Sora,sans-serif; font-size:36px; margin-bottom:4px;'>
            🎓 Student Performance Predictor
        </h1>
        <p style='color:rgba(255,255,255,0.5); font-size:15px; margin-bottom:32px;'>
            Fill in your details on the left and hit <b>Predict</b> to see your score.
        </p>
    """, unsafe_allow_html=True)

    # ── Sidebar inputs ──
    with st.sidebar:
        st.markdown("""
            <h2 style='font-family:Sora,sans-serif; font-size:20px; margin-bottom:20px;'>
                📋 Your Details
            </h2>
        """, unsafe_allow_html=True)

        hour_studied = st.slider("📚 Hours Studied per Day", 1, 10, 5)
        previous_score = st.number_input("📝 Previous Exam Score", min_value=40, max_value=100, value=70)
        extra = st.selectbox("🏆 Extracurricular Activity", ["Yes", "No"])
        sleeping_hour = st.slider("😴 Sleep Hours per Night", 4, 10, 7)
        number_of_paper_solved = st.slider("📄 Question Papers Practiced", 0, 10, 5)

        st.markdown("---")
        predict_btn = st.button("🔍 Predict My Score", use_container_width=True)

    # ── Main area ──
    if predict_btn:
        user_data = {
            "Hours Studied": hour_studied,
            "Previous Scores": previous_score,
            "Extracurricular Activities": extra,
            "Sleep Hours": sleeping_hour,
            "Sample Question Papers Practiced": number_of_paper_solved
        }

        with st.spinner("Calculating your performance..."):
            prediction = predict_data(user_data)
            score = round(float(prediction[0]), 2)
            score_clamped = max(0, min(100, score))

        grade, emoji, color = get_grade(score_clamped)

        # ── Score card ──
        st.markdown(f"""
            <div class="score-card">
                <div class="score-number">{score_clamped}</div>
                <div class="score-label">Predicted Performance Score</div>
                <div style="margin-top:16px; font-size:28px; font-family:'Sora',sans-serif; color:{color};">
                    {emoji} Grade: {grade}
                </div>
            </div>
        """, unsafe_allow_html=True)

        # ── Metrics ──
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📚 Study Hours", f"{hour_studied}h")
        col2.metric("😴 Sleep Hours", f"{sleeping_hour}h")
        col3.metric("📄 Papers Done", number_of_paper_solved)
        col4.metric("📝 Prev. Score", previous_score)

        st.markdown("---")

        # ── Progress bar ──
        st.markdown("### 📊 Score Progress")
        st.progress(int(score_clamped / 100 * 100))

        st.markdown("---")

        # ── Tips ──
        st.markdown("### 💡 Personalized Tips")
        tips = get_tips(hour_studied, sleeping_hour, number_of_paper_solved, extra)
        for tip in tips:
            st.markdown(f'<div class="tip-card">{tip}</div>', unsafe_allow_html=True)

        st.markdown("---")

        # ── Input summary ──
        st.markdown("### 📋 Your Input Summary")
        st.markdown(f"""
        <table style="width:100%; border-collapse:collapse; font-family:'DM Sans',sans-serif; font-size:14px;">
            <thead>
                <tr style="background:rgba(99,102,241,0.45);">
                    <th style="padding:12px 16px; color:#ffffff; font-family:'Sora',sans-serif; font-weight:700; text-align:left; border-bottom:2px solid rgba(99,102,241,0.7);">📚 Hours Studied</th>
                    <th style="padding:12px 16px; color:#ffffff; font-family:'Sora',sans-serif; font-weight:700; text-align:left; border-bottom:2px solid rgba(99,102,241,0.7);">📝 Previous Score</th>
                    <th style="padding:12px 16px; color:#ffffff; font-family:'Sora',sans-serif; font-weight:700; text-align:left; border-bottom:2px solid rgba(99,102,241,0.7);">🏆 Extracurricular</th>
                    <th style="padding:12px 16px; color:#ffffff; font-family:'Sora',sans-serif; font-weight:700; text-align:left; border-bottom:2px solid rgba(99,102,241,0.7);">😴 Sleep Hours</th>
                    <th style="padding:12px 16px; color:#ffffff; font-family:'Sora',sans-serif; font-weight:700; text-align:left; border-bottom:2px solid rgba(99,102,241,0.7);">📄 Papers Practiced</th>
                    <th style="padding:12px 16px; color:#ffffff; font-family:'Sora',sans-serif; font-weight:700; text-align:left; border-bottom:2px solid rgba(99,102,241,0.7);">🎯 Predicted Score</th>
                    <th style="padding:12px 16px; color:#ffffff; font-family:'Sora',sans-serif; font-weight:700; text-align:left; border-bottom:2px solid rgba(99,102,241,0.7);">🏅 Grade</th>
                </tr>
            </thead>
            <tbody>
                <tr style="background:rgba(255,255,255,0.04);">
                    <td style="padding:12px 16px; color:#ffffff;">{hour_studied}h</td>
                    <td style="padding:12px 16px; color:#ffffff;">{previous_score}</td>
                    <td style="padding:12px 16px; color:#ffffff;">{extra}</td>
                    <td style="padding:12px 16px; color:#ffffff;">{sleeping_hour}h</td>
                    <td style="padding:12px 16px; color:#ffffff;">{number_of_paper_solved}</td>
                    <td style="padding:12px 16px; color:#ffffff; font-weight:600;">{score_clamped}</td>
                    <td style="padding:12px 16px; color:{color}; font-weight:700;">{emoji} {grade}</td>
                </tr>
            </tbody>
        </table>
        """, unsafe_allow_html=True)

    else:
        # Welcome state (before prediction)
        st.markdown("""
            <div class="welcome-card">
                <div style="font-size:64px; margin-bottom:16px;">🎓</div>
                <h2>Ready to predict your performance?</h2>
                <p>Enter your study details in the sidebar and click <b>Predict My Score</b>.</p>
            </div>
        """, unsafe_allow_html=True)

    # ── Footer ──
    st.markdown("""
        <div class="footer">
            Made by <a href="#"> T Janani</a> &nbsp;|&nbsp;
            📧 <a href="mailto:Jananiravi2175@gmail.com">Jananiravi2175@gmail.com</a>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
