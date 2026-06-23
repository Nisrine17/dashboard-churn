# ============================================
# DASHBOARD CHURN - MOBILE MONEY
# Fichier : streamlit_app.py
# Lancement : streamlit run streamlit_app.py
# ============================================
# Version 2 - 23 Juin 2026
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURATION DE LA PAGE
# ============================================
st.set_page_config(
    page_title="Dashboard Churn Mobile Money",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CHARGEMENT DES DONNÉES
# ============================================
@st.cache_data
def load_data():
    df = pd.read_csv('sample_data.csv', sep=';')
    
    # Nettoyage
    df['typetransaction'] = df['typetransaction'].str.upper().str.strip()
    df['channel'] = df['channel'].fillna('AUTRE')
    df['statuscode'] = df['statuscode'].fillna('INCONNU')
    df['churn'] = df['churn'].map({'True': 1, 'False': 0, True: 1, False: 0}).fillna(0).astype(int)
    
    # Montants
    df = df[df['montant'] <= 10_000_000]
    df = df[df['montant'] > 0]
    
    # Date
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y %H:%M', errors='coerce')
    df = df.dropna(subset=['date'])
    df['heure'] = df['date'].dt.hour
    df['jour_semaine'] = df['date'].dt.weekday
    df['est_weekend'] = (df['jour_semaine'] >= 5).astype(int)
    
    return df

@st.cache_resource
def load_model():
    try:
        return joblib.load('modele_churn.pkl')
    except:
        return None

# Chargement
df = load_data()
model = load_model()

# ============================================
# SIDEBAR - MENU
# ============================================
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Choisir une page",
    [
        "🏠 Vue d'ensemble",
        "📈 Analyse du churn",
        "🎯 Segmentation risque",
        "🔮 Prédiction",
        "🤖 Performance modèle",
        "💡 Recommandations",
        "📊 Données brutes"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Filtres")

channels = st.sidebar.multiselect("Canal", df['channel'].unique(), default=df['channel'].unique())
types = st.sidebar.multiselect("Type transaction", df['typetransaction'].unique(), default=df['typetransaction'].unique()[:5])

df_filtered = df[df['channel'].isin(channels) & df['typetransaction'].isin(types)]

# ============================================
# PAGE 1 : VUE D'ENSEMBLE
# ============================================
if page == "🏠 Vue d'ensemble":
    st.title("📱 Dashboard Churn Mobile Money")
    st.markdown("*Analyse et prédiction du churn client*")
    st.markdown("---")

    # KPI CALCULS
    taux_churn = df_filtered['churn'].mean() * 100
    transactions = len(df_filtered)
    montant_moyen = df_filtered['montant'].mean()
    volume_total = df_filtered['montant'].sum()
    fees_moyens = df_filtered['fees'].mean()
    perte_churn = df_filtered.loc[df_filtered['churn'] == 1, 'montant'].sum()

    # STYLE CARTES
    st.markdown("""
    <style>
    .kpi-card {
        background: linear-gradient(135deg, #0f172a, #1e293b);
        padding: 18px;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 10px 25px rgba(0,0,0,0.35);
        border: 1px solid rgba(255,255,255,0.08);
        transition: all 0.25s ease;
        margin-bottom: 15px;
    }
    .kpi-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 14px 35px rgba(0,0,0,0.45);
    }
    .kpi-title {
        font-size: 13px;
        color: #94a3b8;
        margin-bottom: 6px;
        letter-spacing: 0.5px;
    }
    .kpi-value {
        font-size: 22px;
        font-weight: 700;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

    # COLONNES (3 CARTES)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">💰 Transactions</div>
            <div class="kpi-value">{transactions:,}</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">📊 Montant moyen</div>
            <div class="kpi-value">{montant_moyen:,.0f} FCFA</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">⚠️ Taux churn</div>
            <div class="kpi-value">{taux_churn:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">💸 Frais moyens</div>
            <div class="kpi-value">{fees_moyens:.2f} FCFA</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">📈 Volume total</div>
            <div class="kpi-value">{volume_total/1_000_000:.1f}M FCFA</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">📉 Perte churn</div>
            <div class="kpi-value">{perte_churn/1_000_000:.1f}M FCFA</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    # Insights automatiques
    st.subheader("💡 Insights automatiques")
    
    top_type = df_filtered.groupby('typetransaction')['churn'].mean().idxmax()
    top_type_rate = df_filtered.groupby('typetransaction')['churn'].mean().max()
    top_channel = df_filtered.groupby('channel')['churn'].mean().idxmax()
    top_channel_rate = df_filtered.groupby('channel')['churn'].mean().max()
    top_hour = df_filtered.groupby('heure')['churn'].mean().idxmax()
    top_hour_rate = df_filtered.groupby('heure')['churn'].mean().max()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"🔴 **Type le plus risqué :** {top_type}\n\n*{top_type_rate:.1%} de churn*")
    with col2:
        st.warning(f"📡 **Canal le plus risqué :** {top_channel}\n\n*{top_channel_rate:.1%} de churn*")
    with col3:
        st.error(f"⏰ **Heure critique :** {top_hour}h\n\n*{top_hour_rate:.1%} de churn*")
    
    st.markdown("---")
    
    # Graphiques
    col1, col2 = st.columns(2)
    
    with col1:
        churn_type = df_filtered.groupby('typetransaction')['churn'].mean().sort_values(ascending=False).head(10)
        fig1 = px.bar(
            x=churn_type.values, y=churn_type.index, orientation='h',
            title="🔴 Top 10 - Taux de churn par type",
            labels={'x': 'Taux churn', 'y': ''},
            color=churn_type.values, color_continuous_scale='RdYlGn_r',
            text=churn_type.apply(lambda x: f"{x:.1%}")
        )
        fig1.update_traces(textposition='outside')
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        churn_canal = df_filtered.groupby('channel')['churn'].mean().sort_values()
        fig2 = px.bar(
            x=churn_canal.values, y=churn_canal.index, orientation='h',
            title="📡 Taux de churn par canal",
            labels={'x': 'Taux churn', 'y': ''},
            color=churn_canal.values, color_continuous_scale='RdYlGn_r',
            text=churn_canal.apply(lambda x: f"{x:.1%}")
        )
        fig2.update_traces(textposition='outside')
        st.plotly_chart(fig2, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        montants_filtres = df_filtered[df_filtered['montant'] < df_filtered['montant'].quantile(0.95)]
        fig3 = px.histogram(
            montants_filtres, x='montant', nbins=50,
            title="📈 Distribution des montants (95%)",
            labels={'montant': 'Montant (FCFA)'},
            color_discrete_sequence=['steelblue']
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        churn_counts = df_filtered['churn'].value_counts()
        fig4 = px.pie(
            values=churn_counts.values, names=['Fidèles', 'Churn'],
            title=f"Répartition du churn ({taux_churn:.1f}%)",
            color_discrete_sequence=['#4facfe', '#ff6b6b'],
            hole=0.4
        )
        st.plotly_chart(fig4, use_container_width=True)

# ============================================
# PAGE 2 : ANALYSE DU CHURN
# ============================================
elif page == "📈 Analyse du churn":
    st.title("📊 Analyse approfondie du churn")
    st.markdown("---")
    
    # Évolution temporelle
    st.subheader("📅 Évolution du churn")
    daily_churn = df_filtered.groupby(df_filtered['date'].dt.date)['churn'].mean().reset_index()
    fig5 = px.line(
        daily_churn, x='date', y='churn',
        title="Taux de churn par jour",
        labels={'date': 'Date', 'churn': 'Taux churn'},
        markers=True, color_discrete_sequence=['#ff6b6b']
    )
    st.plotly_chart(fig5, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        hour_churn = df_filtered.groupby('heure')['churn'].mean()
        fig6 = px.bar(
            x=hour_churn.index, y=hour_churn.values,
            title="⏰ Taux de churn par heure",
            labels={'x': 'Heure', 'y': 'Taux churn'},
            color=hour_churn.values, color_continuous_scale='RdYlGn_r'
        )
        fig6.add_hline(y=df_filtered['churn'].mean(), line_dash="dash", line_color="red")
        st.plotly_chart(fig6, use_container_width=True)
    
    with col2:
        weekend_churn = df_filtered.groupby('est_weekend')['churn'].mean()
        fig7 = px.pie(
            values=weekend_churn.values, names=['Semaine', 'Weekend'],
            title="📆 Churn : Weekend vs Semaine",
            color_discrete_sequence=['#4facfe', '#ff6b6b'],
            hole=0.4
        )
        st.plotly_chart(fig7, use_container_width=True)
    
    # Heatmap
    st.subheader("🗺️ Heatmap du churn (Heure × Jour)")
    heatmap_data = df_filtered.groupby(['heure', 'jour_semaine'])['churn'].mean().unstack()
    fig8 = px.imshow(
        heatmap_data,
        title="Taux de churn par heure et jour de semaine",
        labels={'x': 'Jour (0=lundi, 6=dimanche)', 'y': 'Heure', 'color': 'Taux churn'},
        color_continuous_scale='RdYlGn_r',
        aspect='auto'
    )
    st.plotly_chart(fig8, use_container_width=True)
    
    # Tableau des types à risque
    st.subheader("⚠️ Top 15 des types de transaction les plus risqués")
    risk_table = df_filtered.groupby('typetransaction').agg({
        'churn': 'mean',
        'montant': 'mean',
        'fees': 'mean',
        'heure': 'mean'
    }).round(2).rename(columns={
        'churn': 'Taux churn',
        'montant': 'Montant moyen',
        'fees': 'Frais moyen',
        'heure': 'Heure moyenne'
    })
    risk_table['Taux churn'] = risk_table['Taux churn'].apply(lambda x: f"{x:.2%}")
    st.dataframe(risk_table.sort_values('Taux churn', ascending=False).head(15), use_container_width=True, height=400)

# ============================================
# PAGE 3 : SEGMENTATION RISQUE
# ============================================
elif page == "🎯 Segmentation risque":
    st.title("🎯 Segmentation des clients par niveau de risque")
    st.markdown("---")
    
    if model is not None:
        df_risk = df_filtered.copy()
        X_risk = df_risk[['typetransaction', 'channel', 'statuscode', 'montant', 'fees', 'heure', 'est_weekend']].copy()
        
        le_type = LabelEncoder()
        le_channel = LabelEncoder()
        le_status = LabelEncoder()
        
        X_risk['typetransaction'] = le_type.fit_transform(X_risk['typetransaction'].astype(str))
        X_risk['channel'] = le_channel.fit_transform(X_risk['channel'].astype(str))
        X_risk['statuscode'] = le_status.fit_transform(X_risk['statuscode'].astype(str))
        
        try:
            probas = model.named_steps['classifier'].predict_proba(X_risk)[:, 1]
        except:
            probas = model.predict_proba(X_risk)[:, 1]
        
        df_risk['proba_churn'] = probas
        df_risk['niveau_risque'] = pd.cut(
            df_risk['proba_churn'],
            bins=[0, 0.3, 0.6, 1],
            labels=['🟢 Faible (<30%)', '🟡 Moyen (30-60%)', '🔴 Élevé (>60%)']
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            risk_counts = df_risk['niveau_risque'].value_counts()
            fig9 = px.pie(
                values=risk_counts.values, names=risk_counts.index,
                title="Répartition par niveau de risque",
                color_discrete_sequence=['#4facfe', '#ffa500', '#ff6b6b'],
                hole=0.4
            )
            st.plotly_chart(fig9, use_container_width=True)
        
        with col2:
            churn_by_risk = df_risk.groupby('niveau_risque', observed=False)['churn'].mean()
            fig10 = px.bar(
                x=churn_by_risk.index, y=churn_by_risk.values,
                title="Taux de churn réel par niveau de risque",
                color=churn_by_risk.values, color_continuous_scale='RdYlGn_r',
                text=churn_by_risk.apply(lambda x: f"{x:.1%}")
            )
            st.plotly_chart(fig10, use_container_width=True)
        
        st.subheader("🔴 Top 20 transactions à risque élevé")
        high_risk = df_risk[df_risk['proba_churn'] > 0.6].nlargest(20, 'proba_churn')
        if len(high_risk) > 0:
            st.dataframe(
                high_risk[['channel', 'typetransaction', 'montant', 'heure', 'proba_churn']],
                use_container_width=True
            )
        else:
            st.info("Aucune transaction à risque élevé détectée")
    else:
        st.error("❌ Modèle non disponible")

# ============================================
# PAGE 4 : PRÉDICTION
# ============================================
elif page == "🔮 Prédiction":
    st.title("🔮 Prédiction du churn")
    st.markdown("Renseignez les informations ci-dessous")
    st.markdown("---")
    
    if model is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            type_trans = st.selectbox("Type de transaction", df['typetransaction'].unique())
            canal = st.selectbox("Canal", df['channel'].unique())
            montant = st.number_input("Montant (FCFA)", min_value=0, value=5000, step=1000)
        
        with col2:
            frais = st.number_input("Frais (FCFA)", min_value=0, value=0, step=100)
            status = st.selectbox("Statut", df['statuscode'].unique())
            heure = st.slider("Heure", 0, 23, 12)
        
        weekend = st.checkbox("Transaction le weekend ?")
        
        if st.button("🔍 Prédire", type="primary"):
            try:
                le_type = LabelEncoder()
                le_channel = LabelEncoder()
                le_status = LabelEncoder()
                
                le_type.fit(df['typetransaction'].unique())
                le_channel.fit(df['channel'].unique())
                le_status.fit(df['statuscode'].unique())
                
                type_encoded = le_type.transform([type_trans])[0]
                channel_encoded = le_channel.transform([canal])[0]
                status_encoded = le_status.transform([status])[0]
                
                input_data = pd.DataFrame({
                    'typetransaction': [type_encoded],
                    'channel': [channel_encoded],
                    'statuscode': [status_encoded],
                    'montant': [montant],
                    'fees': [frais],
                    'heure': [heure],
                    'est_weekend': [1 if weekend else 0]
                })
                
                try:
                    proba = model.named_steps['classifier'].predict_proba(input_data)[0][1]
                except:
                    proba = model.predict_proba(input_data)[0][1]
                    
                prediction = 1 if proba > 0.3 else 0
                
                st.markdown("---")
                col1, col2 = st.columns(2)
                
                with col1:
                    if prediction == 1:
                        st.error(f"⚠️ RISQUE ÉLEVÉ de churn")
                        st.metric("Probabilité", f"{proba:.1%}")
                    else:
                        st.success(f"✅ RISQUE FAIBLE de churn")
                        st.metric("Probabilité", f"{(1-proba)*100:.1f}%")
                
                with col2:
                    niveau = "🔴 Élevé" if proba > 0.6 else "🟡 Moyen" if proba > 0.3 else "🟢 Faible"
                    st.info(f"**Niveau de risque :** {niveau}")
                    
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=proba * 100,
                        title={"text": "Risque de churn (%)"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkred" if proba > 0.5 else "darkgreen"},
                            'steps': [
                                {'range': [0, 30], 'color': "lightgreen"},
                                {'range': [30, 70], 'color': "orange"},
                                {'range': [70, 100], 'color': "salmon"}
                            ]
                        }
                    ))
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Erreur : {e}")
    else:
        st.error("❌ Modèle non chargé")

# ============================================
# PAGE 5 : PERFORMANCE MODÈLE
# ============================================
elif page == "🤖 Performance modèle":
    st.title("🤖 Performance du modèle de prédiction")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Métriques du modèle")
        st.markdown("""
        | Métrique | Valeur |
        |----------|--------|
        | **Accuracy** | 61.27% |
        | **Recall Churn** | 44% |
        | **Précision Churn** | 10% |
        | **F1-Score** | 16% |
        """)
    
    with col2:
        st.subheader("🎯 Importance des variables")
        try:
            if model is not None:
                try:
                    importances = model.named_steps['classifier'].feature_importances_
                except:
                    importances = model.feature_importances_
                features = ['montant', 'heure', 'fees', 'typetransaction', 'channel', 'est_weekend', 'statuscode']
                importance_data = pd.DataFrame({
                    'Variable': features,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
            else:
                importance_data = pd.DataFrame({
                    'Variable': ['montant', 'heure', 'fees', 'typetransaction', 'channel', 'est_weekend', 'statuscode'],
                    'Importance': [54.65, 23.95, 8.57, 5.80, 4.92, 1.66, 0.45]
                })
        except:
            importance_data = pd.DataFrame({
                'Variable': ['montant', 'heure', 'fees', 'typetransaction', 'channel', 'est_weekend', 'statuscode'],
                'Importance': [54.65, 23.95, 8.57, 5.80, 4.92, 1.66, 0.45]
            })
        
        fig11 = px.bar(
            importance_data, x='Importance', y='Variable', orientation='h',
            title="Importance des variables",
            color='Importance', color_continuous_scale='Viridis',
            text=importance_data['Importance'].apply(lambda x: f"{x:.1f}%")
        )
        st.plotly_chart(fig11, use_container_width=True)
    
    st.subheader("📈 Matrice de confusion")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("✅ Vrais Fidèles", "103 478")
        st.caption("Bien classifiés")
    
    with col2:
        st.metric("❌ Faux Churn", "61 126")
        st.caption("Fidèles → Churn")
    
    with col3:
        st.metric("🎯 Vrais Churn", "6 536")
        st.caption("Bien détectés")
    
    st.info("""
    **💡 Interprétation :**
    - Le modèle détecte 44% des vrais churn
    - Il y a 61 126 faux positifs (clients fidèles signalés à tort)
    - Le modèle est utile pour une première détection mais doit être affiné
    """)

# ============================================
# PAGE 6 : RECOMMANDATIONS
# ============================================
elif page == "💡 Recommandations":
    st.title("💡 Recommandations stratégiques")
    st.markdown("---")
    
    top_channel_rec = df_filtered.groupby('channel')['churn'].mean().idxmax()
    top_channel_rate_rec = df_filtered.groupby('channel')['churn'].mean().max()
    
    st.markdown(f"""
    ### 🎯 Actions à mettre en place
    
    | Priorité | Recommandation | Impact attendu |
    |----------|----------------|----------------|
    | **Haute** | Renforcer la fidélisation sur le canal **{top_channel_rec}** (taux: {top_channel_rate_rec:.1%}) | Réduction du churn de 2-3% |
    | **Haute** | Cibler les clients faisant des **TRANSFERT_SOUS_COMPTE** | Actions personnalisées |
    | **Moyenne** | Surveiller les transactions en fin de journée (20h-23h) | Prévention proactive |
    | **Moyenne** | Mettre en place des alertes pour les montants > 100k FCFA | Détection des risques |
    | **Basse** | Réentraîner le modèle mensuellement | Maintien des performances |
    
    ### 📊 Actions de fidélisation
    
    1. **Pour les clients à risque élevé (>60%)**
       - Appel téléphonique personnalisé
       - Offre de bienvenue / réduction
    
    2. **Pour les clients à risque moyen (30-60%)**
       - SMS de rappel
       - Notification push
    
    3. **Pour les clients à risque faible (<30%)**
       - Programme de fidélité standard
    
    ### 🔄 Améliorations du modèle
    
    - Collecter plus de données historiques
    - Tester XGBoost / LightGBM
    - Utiliser SMOTE pour meilleur équilibrage
    """)

# ============================================
# PAGE 7 : DONNÉES BRUTES
# ============================================
elif page == "📊 Données brutes":
    st.title("📊 Données brutes")
    st.markdown("---")
    
    st.dataframe(df_filtered.head(100), use_container_width=True)
    
    csv = df_filtered.to_csv(index=False)
    st.download_button(
        "📥 Télécharger les données (CSV)",
        csv,
        "donnees_filtrees.csv",
        "text/csv"
    )

# ============================================
# FOOTER
# ============================================
st.sidebar.markdown("---")
st.sidebar.info(
    """
    **📊 Dashboard Churn Mobile Money**
    
    🔍 Analyse du churn
    
    🔮 Prédiction du risque
    
    🎯 Aide à la décision
    
    🚀 Projet Data Science
    """
)
