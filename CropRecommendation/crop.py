import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load image
image = Image.open("crop_image.jpg")

# Set page layout
st.set_page_config(layout="wide")

# Streamlit UI
st.title("🌱 Smart Farming Assistant - Crop Recommendation")

# Layout with three columns (middle one is spacer)
col1, spacer, col2 = st.columns([3, 0.3, 5])

with col1:
    st.header("🔍 Input Soil & Weather Conditions")
    st.write("Enter the values below to get the best crop recommendation.")
    
    nitrogen = st.number_input("🌿 Nitrogen (N)", min_value=0, max_value=100, value=0)
    phosphorus = st.number_input("💧 Phosphorus (P)", min_value=0, max_value=200, value=0)
    potassium = st.number_input("🌾 Potassium (K)", min_value=0, max_value=200, value=0)
    temperature = st.number_input("🌡️ Temperature (°C)", min_value=0.0, max_value=50.0, value=0.0)
    humidity = st.number_input("💦 Humidity (%)", min_value=0.0, max_value=100.0, value=0.0)
    ph = st.number_input("⚖️ pH Level", min_value=0.0, max_value=14.0, value=0.0)
    rainfall = st.number_input("☔ Rainfall (mm)", min_value=0.0, max_value=300.0, value=0.0)

    # Train model every run
    @st.cache_resource  # Caches model to avoid re-training multiple times in a session
    def train_model():
        df = pd.read_csv("crop_dataset.csv")
        X = df.drop(columns=['label'])
        y = df['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        accuracy = accuracy_score(y_test, model.predict(X_test))
        return model, accuracy

    model, accuracy = train_model()

    # Get algorithm name dynamically
    algorithm_name = type(model).__name__

    # Predict button
    if st.button("🌱 Predict Crop", use_container_width=True):
        # Check if all input values are zero
        if nitrogen == 0 and phosphorus == 0 and potassium == 0 and temperature == 0.0 and humidity == 0.0 and ph == 0.0 and rainfall == 0.0:
            st.session_state.predicted_crop = "⚠️ Invalid Input! Please enter valid values."
        else:
            features = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
            prediction = model.predict(features)[0]

            if prediction in model.classes_:  # Ensure prediction is a valid crop
                st.session_state.predicted_crop = prediction
            else:
                st.session_state.predicted_crop = "No suitable crop found. Adjust inputs."

# Middle spacer column (creates vertical divider)
with spacer:
    st.markdown("<div style='border-left: 4px solid none; height: 500px;'></div>", unsafe_allow_html=True)

# Align image and result closely on the right without unnecessary spacing
with col2:
    st.image(image, caption="🌍 Smart Farming", width=550)

    # Display the algorithm used dynamically
    st.markdown(f"**📌 Current Algorithm: {algorithm_name}**", unsafe_allow_html=True)
    st.markdown(f"**📊 Model Accuracy: {accuracy:.2%}**", unsafe_allow_html=True)

    if "predicted_crop" in st.session_state:
        st.subheader("🌾 Recommended Crop:")
        if "⚠️" in st.session_state.predicted_crop:
            st.warning(st.session_state.predicted_crop)
        else:
            st.success(st.session_state.predicted_crop)
            st.write("✅ This crop is best suited for your given soil and climate conditions.")
