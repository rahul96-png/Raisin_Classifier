import os
import pickle
import pandas as pd
import numpy as np
import streamlit as st

# Set page config once at the very top
st.set_page_config(
    page_title="Raisin Classifier",
    page_icon="🍇",
    layout="centered"
)

# Load the model
model_path = "raisin_model.pkl"
model = None
if os.path.exists(model_path):
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading model: {e}")
else:
    st.warning("⚠️ Trained model not found! Please run the training notebook to generate the model. Falling back to heuristic classifier.")

# Heuristic classifier (fallback)
def heuristic_predict(features_df):
    area_val = features_df.loc[0, 'Area']
    if area_val > 80000:
        return "Kecimen"
    else:
        return "Besni"

st.title("🍇 Raisin Type Classifier")
st.write("Enter values to predict Kecimen or Besni raisin varieties.")

# Form-based input to avoid multiple recalculations on every keystroke
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        area = st.number_input("Area", min_value=0.0, value=75000.0)
        major_axis = st.number_input("Major Axis Length", min_value=0.0, value=400.0)
        minor_axis = st.number_input("Minor Axis Length", min_value=0.0, value=240.0)
        eccentricity = st.number_input("Eccentricity", min_value=0.0, max_value=1.0, value=0.7)
    with col2:
        convex_area = st.number_input("Convex Area", min_value=0.0, value=79000.0)
        extent = st.number_input("Extent", min_value=0.0, max_value=1.0, value=0.7)
        perimeter = st.number_input("Perimeter", min_value=0.0, value=1100.0)
        
    submit_button = st.form_submit_button("Predict")

if submit_button:
    # Construct DataFrame to match the features used for training
    features_df = pd.DataFrame([{
        'Area': area,
        'MajorAxisLength': major_axis,
        'MinorAxisLength': minor_axis,
        'Eccentricity': eccentricity,
        'ConvexArea': convex_area,
        'Extent': extent,
        'Perimeter': perimeter
    }])
    
    if model is not None:
        try:
            result = model.predict(features_df)[0]
            st.success(f"🍇 Predicted Type: **{result}** (using Random Forest Classifier)")
        except Exception as e:
            st.error(f"Error predicting with model: {e}")
            result = heuristic_predict(features_df)
            st.info(f"🍇 Fallback Predicted Type: **{result}** (using heuristic)")
    else:
        result = heuristic_predict(features_df)
        st.info(f"🍇 Fallback Predicted Type: **{result}** (using heuristic)")