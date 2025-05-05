import streamlit as st
import numpy as np
import joblib
import requests

# -------------------- Load model and scaler --------------------
model = joblib.load("cotton_crop_model.pkl")
scaler = joblib.load("scaler.pkl")

# -------------------- Crop dictionary --------------------
crop_dict = {
    'rice': 1, 'maize': 2, 'jute': 3, 'cotton': 4, 'coconut': 5, 'papaya': 6,
    'orange': 7, 'apple': 8, 'muskmelon': 9, 'watermelon': 10, 'grapes': 11,
    'mango': 12, 'banana': 13, 'pomegranate': 14, 'lentil': 15, 'blackgram': 16,
    'mungbean': 17, 'mothbeans': 18, 'pigeonpeas': 19, 'kidneybeans': 20,
    'chickpea': 21, 'coffee': 22
}
reverse_crop_dict = {v: k for k, v in crop_dict.items()}

# -------------------- Prediction Function --------------------
def predict_crop(model, scaler, temperature, humidity, ph, rainfall):
    features = np.array([[temperature, humidity, ph, rainfall]])
    scaled = scaler.transform(features)
    predicted_label = model.predict(scaled)[0]
    predicted_crop = reverse_crop_dict.get(predicted_label, "Unknown")
    return predicted_crop

# -------------------- ThingSpeak Fetch --------------------
def get_latest_thingspeak_data(channel_id, read_api_key):
    url = f"https://api.thingspeak.com/channels/{channel_id}/feeds.json?results=1&api_key={read_api_key}"
    try:
        response = requests.get(url)
        data = response.json()
        feeds = data["feeds"]
        if feeds:
            return feeds[0]
        else:
            return None
    except Exception as e:
        st.error(f"❌ Error fetching data: {e}")
        return None

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Crop Prediction", layout="centered")
st.title("🌾 Crop Prediction App")

st.write("Enter the environmental conditions or fetch from ThingSpeak to predict the most suitable crop.")

# Input fields
col1, col2 = st.columns(2)
with col1:
    temperature = st.number_input("🌡️ Temperature (°C)", step=0.1, format="%.2f")
    ph = st.number_input("🧪 Soil pH", step=0.01, format="%.2f")
with col2:
    humidity = st.number_input("💧 Humidity (%)", step=0.1, format="%.2f")
    rainfall = st.number_input("🌧️ Rainfall (mm)", step=0.1, format="%.2f")

# ThingSpeak Integration
st.markdown("---")
if st.button("🔄 Fetch from ThingSpeak"):
    channel_id = 2922258
    read_api_key = "2YNU92QLNQCX3CI0"
    feed = get_latest_thingspeak_data(channel_id, read_api_key)

    if feed:
        try:
            temperature = float(feed.get('field1', 0))
            humidity = float(feed.get('field2', 0))
            ph = float(feed.get('field3', 0))
            rainfall = float(feed.get('field4', 0))

            st.success("✅ Fetched Data:")
            st.write(f"Temperature = {temperature} °C")
            st.write(f"Humidity = {humidity} %")
            st.write(f"pH = {ph}")
            st.write(f"Rainfall = {rainfall} mm")
        except:
            st.error("❌ Error parsing ThingSpeak data.")
    else:
        st.error("❌ No data received from ThingSpeak.")

# Prediction
st.markdown("---")
if st.button("🚀 Predict Crop"):
    try:
        if None in [temperature, humidity, ph, rainfall]:
            st.error("❌ Please input all values or fetch data before predicting.")
        else:
            predicted_crop = predict_crop(
                model, scaler, temperature, humidity, ph, rainfall
            )

            st.subheader(f"🌾 Predicted Crop: **{predicted_crop.capitalize()}**")

    except Exception as e:
        st.error(f"❌ Prediction Error: {e}")
