import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import shap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import time

# -------------------- Page Config --------------------
st.set_page_config(
    page_title="Rock Vs Mine Prediction System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- CSS Styling --------------------
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
h1, h2, h3 { color: #00F5FF !important; }
/* Text */
p, label, span { color: white !important; }
/* Tabs */
button[data-baseweb="tab"] { color: white !important; font-weight: bold; }
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

# -------------------- Load Data --------------------
sonar_data = pd.read_csv('sonar_dataset.csv', header=None)
X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.1, stratify=Y, random_state=25
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

# -------------------- Mobile Detection --------------------
# Simple screen width detection for layout
try:
    screen_width = st.experimental_get_query_params().get("width", [1024])[0]
    is_mobile = int(screen_width) < 768
except:
    is_mobile = False

# -------------------- Sidebar Controls --------------------
with st.sidebar:
    st.header("Signal Controls")
    st.subheader("🌊 Sonar Wave Generator")
    amplitude = st.slider("Signal strength", 0.1, 1.0, 0.5)
    frequency = st.slider("Wave Frequency", 0.1, 10.0, 3.0)
    noise = st.slider("Ocean Noise", 0.0, 0.5, 0.05)

    st.markdown("---")
    if st.button("Predict"):
        st.session_state.predict = True
    else:
        st.session_state.predict = False

    if st.button("🌊 Scan Ocean"):
        st.session_state.input_data = np.random.rand(60)
        st.session_state.scan = True

if "input_data" not in st.session_state:
    x = np.linspace(0, 10, 60)
    wave = amplitude * np.sin(frequency * x)
    noise_signal = np.random.normal(0, noise, 60)
    input_data = np.clip(wave + noise_signal, 0, 1)
    st.session_state.input_data = input_data
    st.session_state.scan = False

input_data = st.session_state.input_data
input_data_reshaped = input_data.reshape(1, -1)
theta = np.linspace(0, 360, 60, endpoint=False)

# -------------------- Signal Waveform --------------------
wave_fig = px.line(x=range(60), y=input_data, title="Live Sonar Signal Waveform")
wave_fig.update_layout(template="plotly_dark")
st.plotly_chart(wave_fig, use_container_width=True)

# -------------------- System Metrics --------------------
if is_mobile:
    st.subheader("📊 System Status")
    st.metric("Signal Strength", f"{amplitude:.2f}")
    st.metric("Wave Frequency", f"{frequency:.2f}")
    st.metric("Ocean Noise", f"{noise:.2f}")
else:
    col1, col2, col3 = st.columns(3)
    col1.metric("Signal Strength", f"{amplitude:.2f}")
    col2.metric("Wave Frequency", f"{frequency:.2f}")
    col3.metric("Ocean Noise", f"{noise:.2f}")

# -------------------- AI Detection --------------------
if st.session_state.predict:
    pred = model.predict(input_data_reshaped)
    prob = model.predict_proba(input_data_reshaped)
    rock_prob = prob[0][list(model.classes_).index("R")]
    mine_prob = prob[0][list(model.classes_).index("M")]

    st.subheader("📡 AI Detection Panel")
    if pred[0] == 'R':
        st.success("✅ SAFE OBJECT DETECTED")
    else:
        st.error("🚨 MINE DETECTED - TAKE ACTION")

    fig = go.Figure(go.Indicator(
        mode='gauge+number',
        value=mine_prob*100,
        title={'text': 'Mine Probability'},
        gauge={'axis': {'range': [0, 100]}}
    ))
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    # Radar chart (first 10 signals)
    radar = go.Figure()
    radar.add_trace(go.Scatterpolar(
        r=input_data[:10],
        theta=[f"S{i+1}" for i in range(10)],
        fill='toself',
        name='Signals'
    ))
    radar.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=False,
        title="Sonar Signal Radar Chart",
        template="plotly_dark"
    )
    st.plotly_chart(radar, use_container_width=True)

# -------------------- SHAP Explainable AI --------------------
try:
    st.subheader("🧠 SHAP Explainable AI")
    explainer = shap.LinearExplainer(model, X_train)
    shap_values = explainer.shap_values(X_test)
    fig2 = px.bar(
        x=[f"S{i}" for i in range(len(shap_values[0]))],
        y=shap_values[0],
        title="Feature Impact on Prediction"
    )
    st.plotly_chart(fig2, use_container_width=True)
except:
    st.warning("SHAP visualization not supported in this environment.")

# -------------------- 3D Ocean Map --------------------
st.subheader("🌊 3D Ocean Minefield Map")
num_points = 20 if is_mobile else 40
x3d = np.random.uniform(-50,50,num_points)
y3d = np.random.uniform(-50,50,num_points)
z3d = np.random.uniform(-20,0,num_points)
labels = np.random.choice(["Rock","Mine"], num_points)
colors = ["green" if l=="Rock" else "red" for l in labels]

fig3d = go.Figure(data=[
    go.Scatter3d(x=x3d, y=y3d, z=z3d, mode='markers',
                 marker=dict(size=6,color=colors), text=labels)
])
fig3d.update_layout(
    template="plotly_dark",
    scene=dict(xaxis_title="Ocean X", yaxis_title="Ocean Y", zaxis_title="Depth"),
    title="3D Ocean Object Map"
)
st.plotly_chart(fig3d, use_container_width=True)

# -------------------- Submarine Navigation --------------------
st.subheader("🚢 Autonomous Submarine Navigation")
sub_x = np.linspace(-50,50,50)
sub_y = np.sin(sub_x/10)*20

fig_nav = go.Figure()
fig_nav.add_trace(go.Scatter(
    x=sub_x, y=sub_y,
    mode="lines+markers",
    name="Submarine Path",
    line=dict(color="cyan", width=4)
))
fig_nav.update_layout(
    template="plotly_dark",
    title="AI Submarine Navigation Route",
    xaxis_title="Ocean X",
    yaxis_title="Ocean Y"
)
st.plotly_chart(fig_nav, use_container_width=True)

# -------------------- Live Radar Sweep --------------------
st.subheader("🚀 Live Sonar Radar Sweep")
chart = st.empty()

if st.session_state.scan:
    for i in range(len(input_data)):
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=input_data,
            theta=theta,
            mode="lines",
            line=dict(color="lime", width=3)
        ))
        fig.add_trace(go.Scatterpolar(
            r=[0,1],
            theta=[theta[i], theta[i]],
            mode="lines",
            line=dict(color="lime", width=5)
        ))
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
        time.sleep(0.05 if not is_mobile else 0.02)

# -------------------- Ocean Mine Tracking --------------------
st.subheader("🛟 Live Submarine Navigation + Mine Tracking")
ocean_chart = st.empty()
num_mines = 8
mine_x = np.random.uniform(-40,40,num_mines)
mine_y = np.random.uniform(-40,40,num_mines)
sub_x_pos = -50
sub_y_pos = 0

for step in range(80):
    sub_x_pos += 1.2
    sub_y_pos = np.sin(sub_x_pos/8)*15
    mine_x += np.random.uniform(-0.3,0.3,num_mines)
    mine_y += np.random.uniform(-0.3,0.3,num_mines)

    fig_ocean = go.Figure()
    fig_ocean.add_trace(go.Scatter(
        x=mine_x, y=mine_y,
        mode="markers",
        marker=dict(size=14,color="red"),
        name="Mines"
    ))
    fig_ocean.add_trace(go.Scatter(
        x=[sub_x_pos], y=[sub_y_pos],
        mode="markers",
        marker=dict(size=18,color="cyan"),
        name="Submarine"
    ))
    fig_ocean.add_shape(
        type="circle",
        xref="x", yref="y",
        x0=sub_x_pos-15, y0=sub_y_pos-15,
        x1=sub_x_pos+15, y1=sub_y_pos+15,
        line=dict(color="lime")
    )
    fig_ocean.update_layout(
        template="plotly_dark",
        title="Live Submarine Navigation + Mine Tracking",
        xaxis_title="Ocean X", yaxis_title="Ocean Y",
        xaxis=dict(range=[-60,60]), yaxis=dict(range=[-60,60])
    )
    ocean_chart.plotly_chart(fig_ocean, use_container_width=True)
    time.sleep(0.15 if not is_mobile else 0.08)
