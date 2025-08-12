from flask import Flask, request, jsonify
import pickle
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# Charger le modèle entraîné
try:
    model = pickle.load(open("car_price_model.pkl", "rb"))
except FileNotFoundError:
    print("Erreur: car_price_model.pkl non trouvé. Assurez-vous d'exécuter car_price_prediction.py d'abord.")
    model = None

@app.route("/", methods=["GET"])
def home():
    return "API de Prédiction du Prix des Voitures. Utilisez l'endpoint /predict pour faire des prédictions."

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Modèle non chargé. Veuillez contacter l'administrateur."}), 500

    data = request.get_json(force=True)

    # Validation des données d'entrée
    required_fields = ["Present_Price", "Driven_kms", "Fuel_Type", "Selling_type", "Transmission", "Owner", "Year"]
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Champ manquant: {field}"}), 400

    try:
        present_price = float(data["Present_Price"])
        driven_kms = float(data["Driven_kms"])
        fuel_type = str(data["Fuel_Type"])
        selling_type = str(data["Selling_type"])
        transmission = str(data["Transmission"])
        owner = int(data["Owner"])
        year = int(data["Year"])
    except ValueError as e:
        return jsonify({"error": f"Erreur de type de données: {e}"}), 400

    # Ingénierie des caractéristiques (doit correspondre à celle utilisée lors de l'entraînement)
    current_year = datetime.now().year
    car_age = current_year - year

    # Créer un DataFrame pour la prédiction
    # Assurez-vous que l'ordre et les noms des colonnes correspondent à ceux du modèle entraîné
    # Les colonnes one-hot encodées doivent être présentes, même si elles sont à zéro
    input_data = pd.DataFrame([[present_price, driven_kms, owner, car_age, 0, 0, 0, 0]],
                               columns=["Present_Price", "Driven_kms", "Owner", "Car_Age",
                                        "Fuel_Type_Diesel", "Fuel_Type_Petrol",
                                        "Selling_type_Individual", "Transmission_Manual"])

    # Gérer l'encodage one-hot pour les types de carburant, de vente et de transmission
    if fuel_type == "Diesel":
        input_data["Fuel_Type_Diesel"] = 1
    elif fuel_type == "Petrol":
        input_data["Fuel_Type_Petrol"] = 1
    # else: 'CNG' ou autre, les deux colonnes resteront à 0, ce qui est correct si le modèle a été entraîné avec drop_first=True

    if selling_type == "Individual":
        input_data["Selling_type_Individual"] = 1

    if transmission == "Manual":
        input_data["Transmission_Manual"] = 1

    try:
        prediction = model.predict(input_data)[0]
        return jsonify({"predicted_price": round(prediction, 2)}), 200
    except Exception as e:
        return jsonify({"error": f"Erreur lors de la prédiction: {e}"}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)


