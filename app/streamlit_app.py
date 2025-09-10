import streamlit as st
import pickle
import pandas as pd
import catboost

def predict_price(df):
    with open("../regression_model/model.pkl", "rb") as f:
        unpickled_model = pickle.load(f)
        print("Model loaded successfully.")
        first_row = df.iloc[0:1] 
        print("first row:",first_row)

        res = unpickled_model.predict(first_row)[0]
        return f"{int(res):,d}".replace(",", ".")

st.title("Property Price Estimator")
st.set_page_config(layout="centered")

@st.dialog("✅ Price Predicted")

def show_prediction(prediction):
    st.markdown(f"The predicted price is: **{prediction}€**")

if "show_prediction" not in st.session_state:

    with st.form("prediction_form"):
        area = st.slider("Property size (m²)*", 10, 700, 100)

        property_type = st.radio(
        "Select the property type*",
        ["House", "Apartment", "Other"],
    )
        zip_code = st.text_input("Enter a zip code*")

        rooms_number = st.number_input(
        "Choose the number of rooms*", value=1, placeholder="Rooms number"
    )
        with st.expander("See optional fields"):
            options = ["A+","A", "B", "C", "D", "E", "F", "G"]
            epc_score = st.selectbox("EPC score", options, index=None)

            options = [ "NEW" , "GOOD" , "TO RENOVATE" , "JUST RENOVATED" , "TO REBUILD"]
            building_state = st.selectbox("Building state", options, index=None)

            land_area = st.number_input(
            "Land area", value=None, placeholder=""
        )
            garden_area = st.number_input(
            "Garden area", value=None, placeholder=""
        )
            full_address = st.text_input("Enter the full address", "")

            terrace_area = st.text_input("Terrace area", "")

            options = ["1", "2","3","4"]
            facades_number = st.selectbox("Number of facades", options, index=None)

            furnished = st.toggle("Furnished")
            open_fire = st.toggle("Open fire")
            terrace = st.toggle("Terrace")
            garden = st.toggle("With garden")
            swimming_pool = st.toggle("With swimming pool")
            equipped_kitchen = st.toggle("Equipped kitchen")



        submitted = st.form_submit_button("Submit")

        if submitted:
            data_dict = {
                "area": area,
                "property-type": property_type,
                "zip-code": zip_code,
                "rooms-number": rooms_number,
                "epc-score": epc_score,
                "building-state": building_state,
                "land-area": land_area,
                "garden-area": garden_area,
                "full-address": full_address,
                "terrace-area": terrace_area,
                "facades-number": facades_number,
                "equipped-kitchen": equipped_kitchen,
                "swimming-pool": swimming_pool,
                "furnished": furnished,
                "open-fire": open_fire,
                "terrace": terrace,
                "garden": garden,
            }
            try:
                if zip_code == '':
                    raise ValueError(
            "Zip code can't be empty. Please provide a valid Belgian postal code."
        )
                pre_processed_data = pre_process_data(data_dict)
                prediction = predict_price(pre_processed_data)
                show_prediction(prediction)

            except Exception as e:
                st.error(f'An error occurred: {e}', icon="🚨")
