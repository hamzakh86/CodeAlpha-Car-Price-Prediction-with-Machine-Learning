import requests
import json

# URL de l'API Flask
API_URL = "http://localhost:5000/predict"

# Données de test pour la prédiction
test_data = {
    "Present_Price": 8.0,
    "Driven_kms": 50000,
    "Fuel_Type": "Petrol",
    "Selling_type": "Dealer",
    "Transmission": "Manual",
    "Owner": 0,
    "Year": 2015
}

def test_api():
    """
    Teste l'API de prédiction du prix des voitures.
    """
    try:
        # Envoyer une requête POST à l'API
        response = requests.post(API_URL, json=test_data)
        
        # Vérifier le statut de la réponse
        if response.status_code == 200:
            result = response.json()
            print("Test réussi!")
            print(f"Données d'entrée: {test_data}")
            print(f"Prix prédit: {result['predicted_price']} lakhs")
        else:
            print(f"Erreur: {response.status_code}")
            print(f"Message: {response.text}")
    
    except requests.exceptions.ConnectionError:
        print("Erreur: Impossible de se connecter à l'API. Assurez-vous que l'application Flask est en cours d'exécution.")
    except Exception as e:
        print(f"Erreur inattendue: {e}")

if __name__ == "__main__":
    print("Test de l'API de prédiction du prix des voitures")
    print("=" * 50)
    test_api()

