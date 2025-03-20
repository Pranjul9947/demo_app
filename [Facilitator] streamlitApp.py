import streamlit as st
import pickle
import numpy as np
import gdown
import os
from PIL import Image

# Set page config
st.set_page_config(page_title="Timelytics", page_icon=":pencil:", layout="wide")

# Display title and description
st.title("Timelytics: Optimize your supply chain with advanced forecasting techniques.")

st.caption(
    "Timelytics is an ensemble model that utilizes three powerful machine learning algorithms - XGBoost, Random Forests, and SVM - to accurately forecast Order to Delivery (OTD) times."
)

# Google Drive File ID (Extract from the shareable link)
file_id = "1SPeDfOLYsCtrsBCWgGG7RyCNnb8zUtQD"

# Define the model filename
model_path = "voting_model.pkl"

# Download model from Google Drive if not already downloaded
if not os.path.exists(model_path):
    with st.spinner("Downloading model from Google Drive..."):
        gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)

# Load the model
try:
    with open(model_path, "rb") as f:
        voting_model = pickle.load(f)
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    voting_model = None

# Define the prediction function
def waitime_predictor(
    purchase_dow,
    purchase_month,
    year,
    product_size_cm3,
    product_weight_g,
    geolocation_state_customer,
    geolocation_state_seller,
    distance,
):
    if voting_model is None:
        st.error("❌ Model not loaded. Please check the model file.")
        return None
    prediction = voting_model.predict(
        np.array(
            [[
                purchase_dow,
                purchase_month,
                year,
                product_size_cm3,
                product_weight_g,
                geolocation_state_customer,
                geolocation_state_seller,
                distance,
            ]]
        )
    )
    return round(prediction[0])

# Sidebar inputs
with st.sidebar:
    img = Image.open("supply_chain_optimisation.jpg")
    st.image(img)
    st.header("Input Parameters")
    purchase_dow = st.number_input("Purchased Day of the Week", min_value=0, max_value=6, step=1, value=3)
    purchase_month = st.number_input("Purchased Month", min_value=1, max_value=12, step=1, value=1)
    year = st.number_input("Purchased Year", value=2018)
    product_size_cm3 = st.number_input("Product Size in cm^3", value=9328)
    product_weight_g = st.number_input("Product Weight in grams", value=1800)
    geolocation_state_customer = st.number_input("Geolocation State of the Customer", value=10)
    geolocation_state_seller = st.number_input("Geolocation State of the Seller", value=20)
    distance = st.number_input("Distance", value=475.35)
    submit = st.button(label="Predict Wait Time!")

# Prediction Output
if submit:
    if voting_model is not None:
        with st.spinner(text="Processing..."):
            prediction = waitime_predictor(
                purchase_dow,
                purchase_month,
                year,
                product_size_cm3,
                product_weight_g,
                geolocation_state_customer,
                geolocation_state_seller,
                distance,
            )
            if prediction is not None:
                st.header("Output: Wait Time in Days")
                st.write(f"🕒 Estimated Delivery Time: **{prediction} days**")
    else:
        st.error("❌ Model not available. Please check the Google Drive link.")
