import streamlit as st
import pickle
import gdown
import numpy as np
import pandas as pd
from PIL import Image

# Set page configuration
st.set_page_config(page_title="Timelytics", page_icon=":pencil:", layout="wide")

st.title("Timelytics: Optimize your supply chain with advanced forecasting techniques.")

st.caption(
    "Timelytics is an ensemble model that utilizes three powerful machine learning algorithms - XGBoost, Random Forests, and Support Vector Machines (SVM) - to accurately forecast Order to Delivery (OTD) times."
)

st.caption(
    "With Timelytics, businesses can identify potential bottlenecks and delays in their supply chain and take proactive measures to address them, reducing lead times and improving delivery times."
)

# Google Drive file ID
file_id = "1SPeDfOLYsCtrsBCWgGG7RyCNnb8zUtQD"
output_path = "voting_model.pkl"

# Load model only when button is clicked
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
    st.session_state.voting_model = None

def load_model():
    gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)
    with open(output_path, "rb") as f:
        st.session_state.voting_model = pickle.load(f)
    st.session_state.model_loaded = True
    st.success("Model loaded successfully!")

# Button to load the model
if st.button("Load Model"):
    load_model()

# Prediction function
def waitime_predictor(
    purchase_dow, purchase_month, year, product_size_cm3, product_weight_g,
    geolocation_state_customer, geolocation_state_seller, distance
):
    model = st.session_state.voting_model
    prediction = model.predict(
        np.array([[purchase_dow, purchase_month, year, product_size_cm3,
                   product_weight_g, geolocation_state_customer,
                   geolocation_state_seller, distance]])
    )
    return round(prediction[0])

# Sidebar for input parameters and sample dataset
with st.sidebar:
    img = Image.open("./assets/supply_chain_optimisation.jpg")
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

    # Sample dataset in sidebar
    st.header("Sample Dataset")
    data = {
        "Purchased DOW": [0, 3, 1],
        "Purchased Month": [6, 3, 1],
        "Purchased Year": [2018, 2017, 2018],
        "Product Size (cm³)": [37206, 63714, 54816],
        "Product Weight (g)": [16250, 7249, 9600],
        "Customer State": [25, 25, 25],
        "Seller State": [20, 7, 20],
        "Distance (km)": [247.94, 250.35, 4.915],
    }
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)

# Output section
with st.container():
    st.header("Output: Wait Time in Days")
    if submit:
        if not st.session_state.model_loaded:
            st.error("Please load the model first!")
        else:
            prediction = waitime_predictor(
                purchase_dow, purchase_month, year, product_size_cm3,
                product_weight_g, geolocation_state_customer,
                geolocation_state_seller, distance
            )
            with st.spinner(text="This may take a moment..."):
                st.write(prediction)
