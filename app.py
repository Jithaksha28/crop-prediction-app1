import streamlit as st
import joblib
import numpy as np
import requests

# Load model and scaler
model = joblib.load('cotton_crop_model.pkl')
scaler = joblib.load('scaler.pkl')

# Crop label mapping
crop_dict = {
    1: 'rice', 2: 'maize', 3: 'jute', 4: 'cotton', 5: 'coconut', 6: 'papaya',
    7: 'orange', 8: 'apple', 9: 'muskmelon', 10: 'watermelon', 11: 'grapes',
    12: 'mango', 13: 'banana', 14: 'pomegranate', 15: 'lentil', 16: 'blackgram',
    17: 'mungbean', 18: 'mothbeans', 19: 'pigeonpeas', 20: 'kidneybeans',
    21: 'chickpea', 22: 'coffee'
}

# Initialize session state for values if not set
for key in ['temperature', 'humidity', 'ph', 'rainfall']:
    if key not in st.session_state:
        st.session_state[key] = None

# Title
st.title("🌾 Cotton Crop Suitability Predictor")

# Sidebar: Input method
st.sidebar.header("📥 Input Method")
input_method = st.sidebar.radio("Choose input type:", ["Manual", "From ThingSpeak"])

# Input Method Logic
if input_method == "Manual":
    st.session_state.temperature = st.number_input("🌡 Temperature (°C)", 0.0, 50.0, 25.0)
    st.session_state.humidity = st.number_input("💧 Humidity (%)", 0.0, 100.0, 70.0)
    st.session_state.ph = st.number_input("🧪 pH", 0.0, 14.0, 6.5)
    st.session_state.rainfall = st.number_input("🌧 Rainfall (mm)", 0.0, 300.0, 100.0)

else:
    st.sidebar.subheader("🔗 ThingSpeak Settings")
    channel_id = st.sidebar.text_input("Channel ID")
    api_key = st.sidebar.text_input("Read API Key")

    if st.sidebar.button("📥 Fetch Data"):
        try:
            url = f"https://api.thingspeak.com/channels/{channel_id}/feeds.json?api_key={api_key}&results=1"
            response = requests.get(url)
            feed = response.json()['feeds'][0]

            st.session_state.temperature = float(feed['field1'])
            st.session_state.humidity = float(feed['field2'])
            st.session_state.ph = float(feed['field5'])
            st.session_state.rainfall = float(feed['field4'])

            st.success(
                f"✅ Fetched Data:\n\n"
                f"🌡 Temperature = {st.session_state.temperature} °C\n"
                f"💧 Humidity = {st.session_state.humidity} %\n"
                f"🧪 pH = {st.session_state.ph}\n"
                f"🌧 Rainfall = {st.session_state.rainfall} mm"
            )

        except Exception as e:
            st.error(f"❌ Error fetching data: {e}")

# Prediction
if st.button("🔍 Predict Crop"):
    t = st.session_state.temperature
    h = st.session_state.humidity
    p = st.session_state.ph
    r = st.session_state.rainfall

    if None in (t, h, p, r):
        st.error("❌ Please input all values or fetch data before predicting.")
    else:
        input_data = np.array([[t, h, p, r]])
        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)[0]
        probabilities = model.predict_proba(scaled_data)[0]

        predicted_crop = crop_dict.get(prediction, "Unknown")
        cotton_prob = round(probabilities[3] * 100, 2)  # Index 3 = cotton

        st.subheader(f"🌱 Recommended Crop: **{predicted_crop.capitalize()}**")
        st.write(f"🧪 Cotton Suitability Probability: **{cotton_prob}%**")

        # Ideal condition check for cotton
        if 21 <= t <= 30 and 50 <= h <= 80 and 6.0 <= p <= 7.5 and 600 <= r <= 1200:
            st.success("✅ Conditions are IDEAL for cotton!")
        elif predicted_crop == "cotton":
            st.warning("⚠️ Conditions aren't ideal, but model still suggests cotton.")
        else:
            st.info(f"Not ideal for cotton. Better for: **{predicted_crop}**")
