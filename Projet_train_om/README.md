# 💻 Prédicteur de Prix d'Ordinateurs Portables

Une application web interactive développée avec **Streamlit** et propulsée par un modèle de Machine Learning (**Random Forest**) pour estimer le prix d'un ordinateur en fonction de ses caractéristiques techniques.

## 🚀 Fonctionnalités
- **Formulaire interactif** : Renseignez la taille de l'écran, la RAM, la vitesse du CPU, le poids et la résolution.
- **Prédiction en temps réel** : Estimation instantanée du prix en euros grâce au modèle pré-entraîné.
- **Interface responsive** : Une mise en page moderne en deux colonnes avec composants Streamlit.

## 🛠️ Technologies utilisées
- **Python** (version 3.14+)
- **Streamlit** (Interface utilisateur)
- **Scikit-Learn** (Modèle RandomForestRegressor)
- **Joblib** (Sauvegarde et chargement du modèle)
- **UV** (Gestionnaire de paquets ultra-rapide)

## 📦 Structure du Projet
```text
├── laptop_data.csv                 # Le jeu de données initial
├── predictiondeprixordinateur.ipynb # Le Notebook d'exploration et d'entraînement
├── random_forest_model.pkl         # Le modèle de Machine Learning exporté
└── app.py                          # Le code de l'application Streamlit