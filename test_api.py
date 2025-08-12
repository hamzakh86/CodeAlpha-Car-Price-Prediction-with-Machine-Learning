import requests
import json

# Flask API URL
API_URL = "http://localhost:5000/predict"

# Test data for prediction
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
    Tests the car price prediction API.
    """
    try:
        # Send a POST request to the API
        response = requests.post(API_URL, json=test_data)

        # Check the response status
        if response.status_code == 200:
            result = response.json()
            print("Test successful!")
            print(f"Input data: {test_data}")
            print(f"Predicted price: {result['predicted_price']} lakhs")
        else:
            print(f"Error: {response.status_code}")
            print(f"Message: {response.text}")

    except requests.exceptions.ConnectionError:
        print("Error: Unable to connect to the API. Make sure the Flask app is running.")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    print("Car Price Prediction API Test")
    print("=" * 50)
    test_api()
