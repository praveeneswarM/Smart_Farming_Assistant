import streamlit as st
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load dataset
data_path = "fertilizer_dataset.csv"
data = pd.read_csv(data_path)

# Ensure column names match expected format
expected_columns = ["Temparature", "Humidity ", "Moisture", "Soil Type", "Crop Type", "Nitrogen", "Potassium", "Phosphorous", "Fertilizer Name"]
data.columns = expected_columns  # Renaming columns

# Encode categorical features
soil_encoder = LabelEncoder()
crop_encoder = LabelEncoder()
fertilizer_encoder = LabelEncoder()

data["Soil Type"] = soil_encoder.fit_transform(data["Soil Type"])
data["Crop Type"] = crop_encoder.fit_transform(data["Crop Type"])
data["Fertilizer Name"] = fertilizer_encoder.fit_transform(data["Fertilizer Name"])

# Define features and target
X = data.drop(columns=["Fertilizer Name"])
y = data["Fertilizer Name"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

# Calculate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Streamlit UI
st.title("🌱 Fertilizer Recommendation System")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📝 Input Parameters")
    temperature = st.number_input("Temperature (°C)", min_value=0, max_value=50, value=0)
    humidity = st.number_input("Humidity (%)", min_value=0, max_value=100, value=0)
    moisture = st.number_input("Moisture (%)", min_value=0, max_value=100, value=0)
    soil = st.selectbox("Select Soil Type", soil_encoder.classes_)
    crop = st.selectbox("Select Crop Type", crop_encoder.classes_)
    nitrogen = st.number_input("Nitrogen Level", min_value=0, max_value=100, value=0)
    phosphorus = st.number_input("Phosphorous Level", min_value=0, max_value=100, value=0)
    potassium = st.number_input("Potassium Level", min_value=0, max_value=100, value=0)

predicted_fertilizer = ""

if st.button("🔍 Recommend Fertilizer"):
    # Check if any input value is 0, which may cause issues
    if temperature == 0 or humidity == 0 or moisture == 0 or nitrogen == 0 or phosphorus == 0 or potassium == 0:
        st.error("Please provide valid input values greater than 0 for all fields.")
    else:
        # Encode input
        soil_encoded = soil_encoder.transform([soil])[0]
        crop_encoded = crop_encoder.transform([crop])[0]

        # Prepare input data
        input_data = pd.DataFrame([[temperature, humidity, moisture, soil_encoded, crop_encoded, nitrogen, potassium, phosphorus]],
                                  columns=["Temparature", "Humidity ", "Moisture", "Soil Type", "Crop Type", "Nitrogen", "Potassium", "Phosphorous"])

        # Predict fertilizer
        prediction_encoded = model.predict(input_data)[0]
        predicted_fertilizer = fertilizer_encoder.inverse_transform([prediction_encoded])[0]

with col2:
    st.subheader("🖼 Fertilizer Guide")
    st.image("fertilizer_image.jpg", caption="Best Fertilizer for Your Crop", use_container_width=True)

    # Display algorithm type and accuracy below the image
    st.write(f" 🤖 Algorithm: {model.__class__.__name__} | 🎯 Accuracy: {accuracy:.2f}")
    st.subheader("✅ Recommendation")
    st.write(f"#### 🏆 Recommended Fertilizer: `{predicted_fertilizer}`")

st.markdown("---")
