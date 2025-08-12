import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from datetime import datetime
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Data Loading and Exploration ---
# Load the dataset
file_path = 'data/car data.csv'
car_data = pd.read_csv(file_path)

print("Initial Data Head:")
print(car_data.head())
print("\nData Info:")
car_data.info()
print("\nMissing Values:")
print(car_data.isnull().sum())
print("\nData Description:")
print(car_data.describe())

# --- 2. Data Preprocessing and Feature Engineering ---
# Feature Engineering: Add 'Car_Age' column
current_year = datetime.now().year
car_data['Car_Age'] = current_year - car_data['Year']

# Drop 'Car_Name' and 'Year' columns (not useful for modeling)
car_data = car_data.drop(['Car_Name', 'Year'], axis=1)

# Encode categorical variables using one-hot encoding
# drop_first=True is used to avoid multicollinearity
car_data_encoded = pd.get_dummies(car_data, columns=['Fuel_Type', 'Selling_type', 'Transmission'], drop_first=True)

print("\nProcessed Data Head:")
print(car_data_encoded.head())

# --- 3. Model Training ---
# Define features (X) and target (y)
X = car_data_encoded.drop('Selling_Price', axis=1)
y = car_data_encoded['Selling_Price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# --- 4. Model Evaluation ---
# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model using R2 score
r2 = r2_score(y_test, y_pred)
print(f"\nR2 Score: {r2}")

# Save the trained model in pickle format
with open('car_price_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("\nModel saved as car_price_model.pkl")

# --- 5. Results Visualization ---
# Create a regression plot to visualize actual vs predicted prices
plt.figure(figsize=(10, 6))
sns.regplot(x=y_test, y=y_pred, scatter_kws={'alpha':0.3})
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price")
plt.grid(True)
plt.savefig('real_vs_predicted_prices.png')
# plt.show() # Commented out to avoid display in a non-graphic environment


