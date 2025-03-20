import streamlit as st
import pickle
import numpy as np
from PIL import Image

# Set the page configuration
st.set_page_config(page_title="Timelytics", page_icon=":pencil:", layout="wide")

# Display the title and captions
st.title("Timelytics: Optimize your supply chain with advanced forecasting techniques.")

st.caption(
    "Timelytics is an ensemble model that utilizes three powerful machine learning algorithms - XGBoost, Random Forests, and Support Vector Machines (SVM) - to accurately forecast Order to Delivery (OTD) times."
)

# File uploader for the model
uploaded_file = st.file_uploader("Upload Model File (voting_model.pkl)", type=["pkl"])

if uploaded_file is not None:
    voting_model = pickle.load(uploaded_file)
    st.success("Model loaded successfully!")

    # Function for making predictions
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
        prediction = voting_model.predict(
            np.array(
                [
                    [
                        purchase_dow,
                        purchase_month,
                        year,
                        product_size_cm3,
                        product_weight_g,
                        geolocation_state_customer,
                        geolocation_state_seller,
                        distance,
                    ]
                ]
            )
        )
        return round(prediction[0])

    # Sidebar input fields
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

    # Prediction Output
    if submit:
        with st.spinner(text="This may take a moment..."):
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
            st.header("Output: Wait Time in Days")
            st.write(prediction)

else:
    st.warning("Please upload a model file to proceed!")
