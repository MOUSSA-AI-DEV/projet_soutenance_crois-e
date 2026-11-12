from pathlib import Path
import sys
import streamlit as st

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.data.load_data import load_data

st.set_page_config(
    page_title="Student Mental Health Risk Analysis",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Analyse et Prédiction du Risque de Santé Mentale des Étudiants")
st.markdown("""
Cette application interactive permet d'explorer les données de santé mentale des étudiants,
de consulter les performances des modèles d'apprentissage automatique et de tester des prédictions en temps réel.
""")

col1, col2, col3 = st.columns(3)

try:
    df = load_data()
    col1.metric("Total Étudiants", df.shape[0])
    col2.metric("Caractéristiques", df.shape[1] - 1)
    col3.metric("Classes de Risque", df["Mental_Health_Risk"].nunique())

    st.subheader("Aperçu du jeu de données")
    st.dataframe(df.head(10), use_container_width=True)
except Exception as e:
    st.error(f"Erreur de chargement des données : {e}")

st.markdown("""
### 🧭 Navigation
- **📊 1_EDA** : Analyse exploratoire, distributions, corrélations et détection des valeurs aberrantes.
- **📈 2_Model_Performance** : Métriques d'évaluation, comparaison des modèles et matrice de confusion.
- **🔮 3_Prediction** : Simulateur de prédiction en direct pour évaluer le niveau de risque d'un étudiant.
""")

