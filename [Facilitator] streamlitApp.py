import streamlit as st
import joblib
import numpy as np
import requests
from io import BytesIO
from PIL import Image

# Set Page Configuration
st.set_page_config(page_title="Timelytics", page_icon=":clock1:", layout="wide")

# Title & Introduction
st.title("🚀 Timelytics: Order-to-Delivery Time Prediction")
st.write(
    "Timelytics is an **AI-powered forecasting tool** that predicts Order-to-Delivery (OTD) time using machine learning models."
)

# Google Drive File ID (Extracted from Link)
file_id = "1SPeDfOLYsCtrsBCWgGG7RyCNnb8zUtQD"
gdrive_url = f"https://drive.google.com/uc?id={file_id}"

# Function to Load Model
@st.cache_resource
def load_model():
    try:
        response = requests.get(gdrive_url)
        response.raise_for_status()
        model = joblib.load(BytesIO(response.content))
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# Load the model
model = load_model()

# Display Success or Error Message
if model is None:
    st.error("❌ Model not available. Please check the Google Drive link.")
else:
    st.success("✅ Model Loaded Successfully!")

# Sidebar for Input Parameters
with st.sidebar:
    st.header("📊 Input Parameters")
    
    # Display Image
    img = Image.open("./assets/supply_chain_optimisation.jpg")
    st.image(img)

    # Input Fields
    purchase_dow = st.number_input("Purchased Day of the Week (0-6)", min_value=0, max_value=6, step=1, value=3)
    purchase_month = st.number_input("Purchased Month (1-12)", min_value=1, max_value=12, step=1, value=1)
    year = st.number_input("Purchased Year", value=2018)
    product_size_cm3 = st.number_input("Product Size in cm³", value=9328)
    product_weight_g = st.number_input("Product Weight in grams", value=1800)
    geolocation_state_customer = st.number_input("Geolocation State of the Customer", value=10)
    geolocation_state_seller = st.number_input("Geolocation State of the Seller", value=20)
    distance = st.number_input("Distance (in km)", value=475.35)

    # Prediction Button
    submit = st.button("📌 Predict Wait Time!")

# Prediction Logic
st.header("📈 Predicted Order-to-Delivery Time")

if submit:
    if model:
        # Prepare input data
        input_data = np.array([[purchase_dow, purchase_month, year, product_size_cm3, product_weight_g,
                                geolocation_state_customer, geolocation_state_seller, distance]])

        # Model Prediction
        prediction = model.predict(input_data)

        # Display Prediction
        st.success(f"🚛 **Estimated Wait Time: {round(prediction[0])} days**")
    else:
        st.error("⚠️ Model not loaded. Please check the Google Drive link.")

# Sample Dataset Display
import pandas as pd

st.header("📂 Sample Dataset")
data = {
    "Purchased Day of the Week": ["0", "3", "1"],
    "Purchased Month": ["6", "3", "1"],
    "Purchased Year": ["2018", "2017", "2018"],
    "Product Size in cm³": ["37206.0", "63714", "54816"],
    "Product Weight in grams": ["16250.0", "7249", "9600"],
    "Geolocation State Customer": ["25", "25", "25"],
    "Geolocation State Seller": ["20", "7", "20"],
    "Distance": ["247.94", "250.35", "4.915"],
}

df = pd.DataFrame(data)
st.write(df)
