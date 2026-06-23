# 📱 Prédiction du Churn Client dans les Services Mobile Money

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://tableau-de-bord-churn-akkhlgmoqvrrnhwxkcnwv4.streamlit.app)

---

## 📖 Contexte

Dans un environnement fortement concurrentiel, la fidélisation des clients constitue un enjeu majeur pour les opérateurs de services Mobile Money. La capacité à identifier les utilisateurs susceptibles de quitter le service permet de mettre en place des actions préventives et d'améliorer la rétention.

Ce projet de Data Science vise à développer un modèle de prédiction du churn à partir de données transactionnelles Mobile Money et à proposer une interface interactive facilitant l'analyse des comportements clients.

---

## 🎯 Objectifs

- Analyser les comportements transactionnels des utilisateurs.
- Identifier les facteurs influençant le churn.
- Construire un modèle de Machine Learning capable de prédire les risques d'attrition.
- Développer un dashboard interactif permettant l'exploration des données et la prédiction en temps réel.

---

## 📊 Description des données

Le jeu de données contient plus de **900 000 transactions** Mobile Money enregistrées sur une période de plusieurs jours.

### Variables principales

| Variable | Description |
|----------|-------------|
| `channel` | Canal utilisé pour la transaction |
| `typetransaction` | Nature de l'opération effectuée |
| `montant` | Montant de la transaction |
| `fees` | Frais associés à l'opération |
| `statuscode` | Statut de la transaction |
| `date` | Date et heure de l'opération |
| `churn` | Variable cible (1 = client perdu, 0 = client actif) |

### 📁 Accès aux données

⚠️ **Les données complètes (`data.csv`) ne sont pas incluses dans ce dépôt** en raison de leur taille (77 Mo).

✅ **Un échantillon représentatif (`sample_data.csv`)** est fourni pour tester l'application et reproduire les analyses.

---

## 🛠️ Technologies utilisées

| Catégorie | Technologies |
|-----------|--------------|
| **Data** | Pandas, NumPy |
| **Machine Learning** | Scikit-Learn (Random Forest, Logistic Regression) |
| **Visualisation** | Plotly |
| **Dashboard** | Streamlit |
| **Déploiement** | Streamlit Cloud |

---

## 🔍 Méthodologie

### 1. Préparation des données
- Nettoyage des données
- Gestion des valeurs manquantes
- Suppression des doublons
- Traitement des valeurs aberrantes

### 2. Analyse exploratoire
- Distribution des montants
- Analyse du churn par canal
- Analyse du churn par type de transaction
- Étude des tendances temporelles

### 3. Modélisation
Deux modèles ont été évalués :
- Régression Logistique
- Random Forest

Le modèle **Random Forest** a présenté les meilleures performances globales et a été retenu pour l'application finale.

---

## 📈 Résultats obtenus

### Performance des modèles

| Modèle | Accuracy | Recall Churn |
|--------|----------|--------------|
| Régression Logistique | 45.15 % | 60 % |
| **Random Forest** | **61.27 %** | **44 %** |

### Variables les plus influentes

1. Montant de la transaction (54.65%)
2. Heure de réalisation (23.95%)
3. Frais appliqués (8.57%)
4. Type de transaction (5.80%)
5. Canal utilisé (4.92%)

---

## 💻 Dashboard Interactif

L'application Streamlit permet :
- La visualisation des indicateurs clés (KPIs)
- L'analyse détaillée du churn (heatmap, évolutions)
- La segmentation des clients par niveau de risque (🔴🟡🟢)
- La consultation des données filtrées
- La prédiction du risque de churn en temps réel

---

## 🚀 Installation et lancement

### Prérequis
- Python 3.8 ou supérieur
- Git

### Cloner le projet

```bash
git clone https://github.com/Nisrine17/dashboard-churn.git
cd dashboard-churn
