import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

print("1. Chargement des données...")
df = pd.read_csv('sample_data.csv', sep=';')

print("2. Nettoyage...")
df['typetransaction'] = df['typetransaction'].str.upper().str.strip()
df['channel'] = df['channel'].fillna('AUTRE')
df['statuscode'] = df['statuscode'].fillna('INCONNU')
df['churn'] = df['churn'].map({'True': 1, 'False': 0, True: 1, False: 0}).fillna(0).astype(int)

df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y %H:%M', errors='coerce')
df['heure'] = df['date'].dt.hour
df['est_weekend'] = (df['date'].dt.weekday >= 5).astype(int)

print("3. Encodage...")
le_type = LabelEncoder()
le_channel = LabelEncoder()
le_status = LabelEncoder()

df['type_enc'] = le_type.fit_transform(df['typetransaction'])
df['channel_enc'] = le_channel.fit_transform(df['channel'])
df['status_enc'] = le_status.fit_transform(df['statuscode'])

print("4. Préparation des features...")
features = ['type_enc', 'channel_enc', 'status_enc', 'montant', 'fees', 'heure', 'est_weekend']
X = df[features]
y = df['churn']

print("5. Imputation...")
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

print("6. Entraînement du modèle...")
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_imputed, y)

print("7. Sauvegarde...")
joblib.dump(model, 'modele_churn.pkl', compress=3)
print("✅ Modèle sauvegardé dans modele_churn.pkl")