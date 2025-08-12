from flask import Flask, request, jsonify
import pickle
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# Load the trained model
try:
    model = pickle.load(open("car_price_model.pkl", "rb"))
except FileNotFoundError:
    print("Error: car_price_model.pkl not found. Make sure to run car_price_prediction.py first.")
    model = None

@app.route("/", methods=["GET"])
def home():
    return "Car Price Prediction API. Use the /predict endpoint to make predictions."

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Please contact the administrator."}), 500

    data = request.get_json(force=True)

    # Input data validation
    required_fields = ["Present_Price", "Driven_kms", "Fuel_Type", "Selling_type", "Transmission", "Owner", "Year"]
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing field: {field}"}), 400

    try:
        present_price = float(data["Present_Price"])
        driven_kms = float(data["Driven_kms"])
        fuel_type = str(data["Fuel_Type"])
        selling_type = str(data["Selling_type"])
        transmission = str(data["Transmission"])
        owner = int(data["Owner"])
        year = int(data["Year"])
    except ValueError as e:
        return jsonify({"error": f"Data type error: {e}"}), 400

    # Feature engineering (must match what was used during training)
    current_year = datetime.now().year
    car_age = current_year - year

    # Create a DataFrame for prediction
    # Ensure that the order and column names match those of the trained model
    # One-hot encoded columns must be present, even if they are zero
    input_data = pd.DataFrame([[present_price, driven_kms, owner, car_age, 0, 0, 0, 0]],
                               columns=["Present_Price", "Driven_kms", "Owner", "Car_Age",
                                        "Fuel_Type_Diesel", "Fuel_Type_Petrol",
                                        "Selling_type_Individual", "Transmission_Manual"])

    # Handle one-hot encoding for fuel, selling, and transmission types
    if fuel_type == "Diesel":
        input_data["Fuel_Type_Diesel"] = 1
    elif fuel_type == "Petrol":
        input_data["Fuel_Type_Petrol"] = 1
    # else: 'CNG' or other, both columns will remain 0, which is correct if the model was trained with drop_first=True

    if selling_type == "Individual":
        input_data["Selling_type_Individual"] = 1

    if transmission == "Manual":
        input_data["Transmission_Manual"] = 1

    try:
        prediction = model.predict(input_data)[0]
        return jsonify({"predicted_price": round(prediction, 2)}), 200
    except Exception as e:
        return jsonify({"error": f"Prediction error: {e}"}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)


