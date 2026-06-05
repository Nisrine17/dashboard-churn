import streamlit as st
import joblib
import numpy as np

# 1. Charger le modèle pré-entraîné
# Il cherche le modèle dans le même dossier que ce fichier app.py
model = joblib.load('random_forest_model.pkl')

# Fonction pour formater les données et faire la prédiction
def predict_price(features):
    # Transforme la liste en tableau NumPy et lui donne la bonne forme (1 ligne, X colonnes)
    features = np.array(features).reshape(1, -1)
    return model.predict(features)[0]

# 2. Interface utilisateur Streamlit
st.set_page_config(page_title="Prédicteur de Prix PC", page_icon="💻", layout="centered")
st.title('💻 Prédiction des Prix des Ordinateurs')
st.write("Modifiez les caractéristiques ci-dessous pour estimer le prix du PC en temps réel.")

st.header('Caractéristiques de l\'ordinateur')

# Configuration des colonnes pour une jolie mise en page côte à côte
col1, col2 = st.columns(2)

with col1:
    inches = st.number_input('Taille de l\'écran (en pouces)', min_value=10.0, max_value=20.0, value=15.6, step=0.1)
    ram = st.selectbox('RAM (en Go)', [4, 8, 12, 16, 24, 32, 64], index=1)
    cpu_ghz = st.number_input('Vitesse du CPU (en GHz)', min_value=0.5, max_value=5.0, value=2.5, step=0.1)

with col2:
    weight = st.number_input('Poids du PC (en kg)', min_value=0.5, max_value=5.0, value=2.0, step=0.1)
    res_x = st.number_input('Résolution Largeur (X)', min_value=800, max_value=4000, value=1920, step=1)
    res_y = st.number_input('Résolution Hauteur (Y)', min_value=600, max_value=3000, value=1080, step=1)

# L'ordre ici doit être EXACTEMENT le même que dans ton X_train lors de l'entraînement
# ['Inches', 'Ram_GB', 'Cpu_GHz', 'Weight_value', 'Resolution_x', 'Resolution_y']
features = [inches, float(ram), cpu_ghz, weight, res_x, res_y]

st.markdown("---")

# Bouton pour lancer le calcul
if st.button('💰 Estimer le prix', use_container_width=True):
    try:
        price = predict_price(features)
        # Affiche le résultat dans un joli encadré vert du succès
        st.success(f"### Le prix estimé pour cette configuration est de : **{price:.2f} €**")
    except Exception as e:
        # En cas de problème (ex: mauvaise forme de features), affiche une alerte rouge
        st.error(f"Erreur lors de la prédiction : {e}")
        
        
        #uv run streamlit run app.py   pour lancer l'application dans le terminal, assurez-vous d'être dans le bon dossier et d'avoir installé les dépendances nécessaires (streamlit, joblib, numpy).
