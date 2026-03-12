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

/* Main app background */
.stApp {
    background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
}

/* Force ALL text to white */
* {
    color: white !important;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background-color: #0E1117 !important;
}

/* Tab text */
button[data-baseweb="tab"] {
    color: white !important;
    font-weight: bold;
}

/* Slider labels */
div[data-baseweb="slider"] label {
    color: white !important;
}

/* Widget labels */
label {
    color: white !important;
}

/* Headers */
h1, h2, h3 {
    color: #00F5FF !important;
}

/* Buttons */
.stButton>button {
    background-color: #00F5FF;
    color: black !important;
    border-radius: 8px;
}

/* Chart containers */
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
st.plotly_chart(wave_fig, use_container_width = True)
if st.sidebar.button("Predict"):
  pred = model.predict(input_data_reshaped)
  prob = model.predict_proba(input_data_reshaped)
  rock_prob = prob[0][list(model.classes_).index("R")]
  mine_prob = prob[0][list(model.classes_).index("M")]
  st.subheader("Prediction Result")
  if(pred[0] == 'R'):
     st.success('Object is rock 🪨')
  else:
     st.error('Object is mine💣')
  fig = go.Figure(go.Indicator(mode = 'gauge + number', value = mine_prob * 100, title = {'text' : 'Mine probability'}, gauge = {'axis' : {'range': [0, 100]}}))
  st.plotly_chart(fig, use_container_width = True)
  radar = go.Figure()
  radar.add_trace(go.Scatterpolar(r = input_data[:10], theta = [f"S{i + 1}" for i in range(10)], fill = 'toself', name = 'Signals'))
  radar.update_layout(polar = dict(radialaxis = dict(visible = True)), showlegend = False, title = "Sonar Signal Radar Chart")
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
tab1, tab2, tab3, tab4 = st.tabs(["🌊 Sonar Sweep","📡 Radar Scanner","🌊 3D Ocean Map","🤖 Autonomous AI"])
with tab1:
    st.subheader("🌊 Real-Time Sonar Sweep")
    theta = np.linspace(0, 360, 60)
    frames = []
    for i in range(60):
      frames.append(
    go.Frame(
        data=[
            go.Scatterpolar(
                r=input_data,
                theta=theta,
                mode='lines',
                line=dict(color='lime', width=3)
            ),
            go.Scatterpolar(
                r=[1],
                theta=[theta[i]],
                mode='markers',
                marker=dict(size=15)
            )
        ]
    )
)
    fig = go.Figure(data=[go.Scatterpolar(r = input_data, theta = theta, mode = "lines", line = dict(color = "lime", width = 3))], frames = frames)
    fig.update_layout(polar = dict(radialaxis = dict(visible = True, range = [0, 1])), showlegend = False, title = "Animated Sonar Radar Sweep", updatemenus = [dict(type = "buttons", buttons = [dict(label = "Start Scan", method = 'animate', args = [None])])])
    st.plotly_chart(fig, use_container_width=True)



