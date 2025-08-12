import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from datetime import datetime
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Chargement et Exploration des Données ---
# Charger le jeu de données
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

# --- 2. Prétraitement des Données et Ingénierie des Caractéristiques ---
# Ingénierie des caractéristiques: Ajouter la colonne 'Car_Age'
current_year = datetime.now().year
car_data['Car_Age'] = current_year - car_data['Year']

# Supprimer les colonnes 'Car_Name' et 'Year' (inutiles pour la modélisation)
car_data = car_data.drop(['Car_Name', 'Year'], axis=1)

# Encoder les variables catégorielles en utilisant l'encodage one-hot
# drop_first=True est utilisé pour éviter la multicolinéarité
car_data_encoded = pd.get_dummies(car_data, columns=['Fuel_Type', 'Selling_type', 'Transmission'], drop_first=True)

print("\nProcessed Data Head:")
print(car_data_encoded.head())

# --- 3. Entraînement du Modèle ---
# Définir les caractéristiques (X) et la cible (y)
X = car_data_encoded.drop('Selling_Price', axis=1)
y = car_data_encoded['Selling_Price']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# Entraîner le modèle de régression linéaire
model = LinearRegression()
model.fit(X_train, y_train)

# --- 4. Évaluation du Modèle ---
# Faire des prédictions sur l'ensemble de test
y_pred = model.predict(X_test)

# Évaluer le modèle en utilisant le score R2
r2 = r2_score(y_test, y_pred)
print(f"\nR2 Score: {r2}")

# Sauvegarder le modèle entraîné au format pickle
with open('car_price_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("\nModel saved as car_price_model.pkl")

# --- 5. Visualisation des Résultats ---
# Créer un graphique de régression pour visualiser les prix réels vs prédits
plt.figure(figsize=(10, 6))
sns.regplot(x=y_test, y=y_pred, scatter_kws={'alpha':0.3})
plt.xlabel("Prix Réel")
plt.ylabel("Prix Prédit")
plt.title("Prix Réel vs Prix Prédit")
plt.grid(True)
plt.savefig('real_vs_predicted_prices.png')
# plt.show() # Commenté pour éviter l'affichage dans un environnement non-graphique


