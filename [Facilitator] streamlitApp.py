import streamlit as st
import joblib
import numpy as np
import pandas as pd
import requests
from io import BytesIO
from PIL import Image

# Set Streamlit page config
st.set_page_config(page_title="Timelytics", page_icon=":pencil:", layout="wide")

st.title("Timelytics: Optimize your supply chain with advanced forecasting techniques.")

st.caption(
    "Timelytics is an ensemble model that utilizes XGBoost, Random Forest, and SVM "
    "to accurately forecast Order to Delivery (OTD) times."
)

# Google Drive model URL
GDRIVE_MODEL_URL = "https://drive.google.com/uc?export=download&id=1SPeDfOLYsCtrsBCWgGG7RyCNnb8zUtQD"

@st.cache_resource
def load_model():
    try:
        response = requests.get(GDRIVE_MODEL_URL)
        response.raise_for_status()  # Raise an error for failed requests
        model = joblib.load(BytesIO(response.content))
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# Load the model
voting_model = load_model()

# Sidebar for input
with st.sidebar:
    img = Image.open("./assets/supply_chain_optimisation.jpg")
    st.image(img)
    st.header("Input Parameters")

    purchase_dow = st.number_input("Purchased Day of the Week", min_value=0, max_value=6, step=1, value=3)
    purchase_month = st.number_input("Purchased Month", min_value=1, max_value=12, step=1, value=1)
    year = st.number_input("Purchased Year", value=2018)
    product_size_cm3 = st.number_input("Product Size in cm³", value=9328)
    product_weight_g = st.number_input("Product Weight in grams", value=1800)
    geolocation_state_customer = st.number_input("Geolocation State of the Customer", value=10)
    geolocation_state_seller = st.number_input("Geolocation State of the Seller", value=20)
    distance = st.number_input("Distance", value=475.35)

    submit = st.button(label="Predict Wait Time!")

# Prediction function
def waitime_predictor(purchase_dow, purchase_month, year, product_size_cm3, product_weight_g, geolocation_state_customer, geolocation_state_seller, distance):
    if voting_model:
        prediction = voting_model.predict(
            np.array([[purchase_dow, purchase_month, year, product_size_cm3, product_weight_g, geolocation_state_customer, geolocation_state_seller, distance]])
        )
        return round(prediction[0])
    else:
        return "❌ Model not available. Please check the Google Drive link."

# Display output
with st.container():
    st.header("Output: Wait Time in Days")

    if submit:
        prediction = waitime_predictor(
            purchase_dow, purchase_month, year, product_size_cm3, product_weight_g, geolocation_state_customer, geolocation_state_seller, distance
        )
        with st.spinner("This may take a moment..."):
            st.write(prediction)

    # Sample dataset
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
    st.header("Sample Dataset")
    st.write(df)
