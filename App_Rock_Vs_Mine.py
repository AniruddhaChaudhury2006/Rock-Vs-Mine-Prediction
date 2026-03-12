import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import shap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
st.set_page_config(page_title = "Rock Vs Mine Prediction System", layout = 'wide')
st.markdown("""<style> body { background-color: #0E1117; } .stApp {  background: linear-gradient(135deg,#0f2027,#203a43,#2c5364); color:white; </style>""", unsafe_allow_html = True)
st.title("🔎 🧠 Futuristic Rock vs Mine AI Detection System")
sonar_data = pd.read_csv('sonar_dataset.csv', header = None)
X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, stratify = Y, random_state = 25)
model = LogisticRegression(max_iter = 1000)
model.fit(X_train, Y_train)
st.sidebar.header("Signal Controls")
auto = st.sidebar.button("Autofill Signals")
if auto:
  default = np.random.rand(60)
else:
  default = np.zeros(60)
input_data = []
for i in range(60):
  val = st.number_input(f"Value {i + 1}")
  input_data.append(val)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
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
st.subheader("🧠 SHAP Explainable AI")
explainer = shap.LinearExplainer(model, X_train)
shap_values = explainer.shap_values(X_test)
fig2 = px.bar(x = [f"S{i}" for i in range(len(shap_values[0]))], y = shap_values[0], title = "Feature Impact on Prediction")
st.plotly_chart(fig2, use_container_width = True)


