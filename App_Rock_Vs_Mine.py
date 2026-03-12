import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import shap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
st.set_page_config(page_title = "Rock Vs Mine Prediction System", layout = 'wide')
st.markdown("""
<style>

/* Main background */
.stApp {
    background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #0E1117 !important;
}

/* Headers */
h1, h2, h3 {
    color: #00F5FF !important;
}

/* Text */
p, label, span, div {
    color: white !important;
}

/* Tabs */
button[data-baseweb="tab"] {
    color: white !important;
    font-weight: bold;
}

/* Buttons */
.stButton>button {
    background-color: #00F5FF;
    color: black !important;
    border-radius: 10px;
    font-weight: bold;
}

/* Metric cards */
[data-testid="metric-container"] {
    background-color: rgba(0,0,0,0.4);
    border-radius: 10px;
    padding: 10px;
}

/* Chart container */
.block-container {
    background: rgba(0,0,0,0.35);
    padding: 20px;
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)
st.title("🔎 🧠 Futuristic Rock vs Mine AI Detection System")
sonar_data = pd.read_csv('sonar_dataset.csv', header = None)
X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, stratify = Y, random_state = 25)
model = LogisticRegression(max_iter = 1000)
model.fit(X_train, Y_train)
st.sidebar.header("Signal Controls")
st.sidebar.subheader("🌊 Sonar Wave Generator")
amplitude = st.sidebar.slider("Signal strength",0.1,1.0,0.5)
frequency = st.sidebar.slider("Wave Frequency",0.1,10.0,3.0)
noise = st.sidebar.slider("Ocean Noise",0.0,0.5,0.05)
x = np.linspace(0, 10, 60)
wave = amplitude * np.sin(frequency * x)
noise_signal = np.random.normal(0, noise, 60)
input_data = wave + noise_signal
input_data = np.clip(input_data, 0, 1)
input_data_reshaped = input_data.reshape(1,-1)
wave_fig = px.line(x = range(60), y = input_data, title = "Live Sonar Signal Waveform")
col1, col2 = st.columns([2, 1])
with col1:
    wave_fig.update_layout(template="plotly_dark")
    st.plotly_chart(wave_fig, use_container_width=True)
    st.subheader("📊 System Status")

    c1, c2, c3 = st.columns(3)

    c1.metric("Signal Strength", f"{amplitude:.2f}")
    c2.metric("Wave Frequency", f"{frequency:.2f}")
    c3.metric("Ocean Noise", f"{noise:.2f}")
with col2:
    st.subheader("📡 AI Detection Panel")
    if st.sidebar.button("Predict"):
       pred = model.predict(input_data_reshaped)
       prob = model.predict_proba(input_data_reshaped)
       rock_prob = prob[0][list(model.classes_).index("R")]
       mine_prob = prob[0][list(model.classes_).index("M")]
       st.subheader("Prediction Result")
       if pred[0] == 'R':
          st.success("✅ SAFE OBJECT DETECTED")
       else:
          st.error("🚨 MINE DETECTED - TAKE ACTION")
       fig = go.Figure(go.Indicator(mode = 'gauge + number', value = mine_prob * 100, title = {'text' : 'Mine probability'}, gauge = {'axis' : {'range': [0, 100]}}))
       fig.update_layout(template="plotly_dark")
       st.plotly_chart(fig, use_container_width = True)
  
       radar = go.Figure()
       radar.add_trace(go.Scatterpolar(r = input_data[:10], theta = [f"S{i + 1}" for i in range(10)], fill = 'toself', name = 'Signals'))
       radar.update_layout(polar = dict(radialaxis = dict(visible = True)), showlegend = False, title = "Sonar Signal Radar Chart")
       radar.update_layout(template="plotly_dark")
       st.plotly_chart(radar, use_container_width = True)
  
try:
  st.subheader("🧠 SHAP Explainable AI")
  explainer = shap.LinearExplainer(model, X_train)
  shap_values = explainer.shap_values(X_test)
  fig2 = px.bar(x = [f"S{i}" for i in range(len(shap_values[0]))], y = shap_values[0], title = "Feature Impact on Prediction")
  st.plotly_chart(fig2, use_container_width = True)
except Exception as e:
  st.warning("SHAP visualization not supported in this environment.")
if st.sidebar.button("🌊 Scan Ocean"):
    input_data = np.random.rand(60)
    input_data_reshaped = input_data.reshape(1,-1)
tab1 = st.tabs(["🌊 Sonar Sweep"])[0]
with tab1:

    st.subheader("🌊 Real-Time Sonar Sweep")

    theta = np.linspace(0, 360, 60, endpoint=False)

    chart = st.empty()

    if st.button("Start Scan"):
        st.session_state.scan = True

    if "scan" in st.session_state and st.session_state.scan:

        for i in range(60):

         fig = go.Figure()

         fig.add_trace(
             go.Scatterpolar(
             r=input_data,
             theta=theta,
             mode="lines",
             line=dict(color="lime", width=3)
        )
    )

        fig.add_trace(
            go.Scatterpolar(
            r=[0,1],
            theta=[theta[i], theta[i]],
            mode="lines",
            line=dict(color="lime", width=5)
        )
    )

        fig.update_layout(
            template="plotly_dark",
            polar=dict(
            bgcolor="black",
            radialaxis=dict(visible=True, range=[0,1], gridcolor="green"),
            angularaxis=dict(gridcolor="green")
        ),
            showlegend=False,
            title="Live Sonar Radar Sweep"
    )

        chart.plotly_chart(fig, use_container_width=True)

        import time
        time.sleep(1)
        

    



