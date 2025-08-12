# Projet de Prédiction du Prix des Voitures

Ce projet implémente un modèle de machine learning pour prédire le prix de vente des voitures d'occasion, basé sur diverses caractéristiques telles que le prix actuel, le kilométrage parcouru, le type de carburant, le type de vente, la transmission et l'âge de la voiture.

## Fichiers du Projet

- `car_price_prediction.py`: Script Python pour le prétraitement des données, l'entraînement du modèle de régression linéaire et la sauvegarde du modèle entraîné.
- `car_price_model.pkl`: Le modèle de régression linéaire entraîné, sauvegardé au format pickle.
- `app.py`: Une application web Flask qui expose une API pour la prédiction du prix des voitures en utilisant le modèle entraîné.
- `data/car data.csv`: Le jeu de données utilisé pour entraîner le modèle.

## Comment Utiliser

1.  **Prérequis**: Assurez-vous d'avoir Python 3.x et les bibliothèques suivantes installées:
    -   `pandas`
    -   `scikit-learn`
    -   `numpy`
    -   `flask`

    Vous pouvez les installer via pip:
    ```bash
    pip install pandas scikit-learn numpy flask
    ```

2.  **Entraînement du Modèle**: Exécutez le script `car_price_prediction.py` pour entraîner le modèle et sauvegarder `car_price_model.pkl`.
    ```bash
    python car_price_prediction.py
    ```

3.  **Lancement de l'API**: Exécutez l'application Flask `app.py`.
    ```bash
    python app.py
    ```
    L'API sera disponible sur `http://0.0.0.0:5000`.

4.  **Faire une Prédiction**: Envoyez une requête POST à l'endpoint `/predict` avec un corps JSON contenant les caractéristiques de la voiture. Exemple:

    ```json
    {
        "Present_Price": 8.0,
        "Driven_kms": 50000,
        "Fuel_Type": "Petrol",
        "Selling_type": "Dealer",
        "Transmission": "Manual",
        "Owner": 0,
        "Year": 2015
    }
    ```

    L'API retournera une prédiction du prix de vente.

## Auteur

Manus (Agent IA)


