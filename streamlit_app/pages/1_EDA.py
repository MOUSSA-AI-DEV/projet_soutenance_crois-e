from pathlib import Path
import sys
import streamlit as st

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.data.load_data import load_data
from src.preprocessing.outliers import detect_outliers
from streamlit_app.components.visualizations import (
    plot_numeric_distributions,
    plot_categorical_distributions,
    plot_correlation_matrix,
    plot_outliers_boxplots,
    plot_target_relationships,
)

st.title("📊 Analyse Exploratoire des Données (EDA)")

df = load_data()

tab1, tab2, tab3, tab4 = st.tabs([
    "Vue Générale",
    "Distributions",
    "Corrélations & Outliers",
    "Relations avec la Cible",
])

with tab1:
    st.subheader("Informations et Qualité des Données")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Lignes", df.shape[0])
    c2.metric("Colonnes", df.shape[1])
    c3.metric("Doublons", int(df.duplicated().sum()))
    c4.metric("Valeurs Manquantes", int(df.isnull().sum().sum()))

    st.write("Statistiques descriptives :")
    st.dataframe(df.describe().T, use_container_width=True)

    st.write("Valeurs manquantes par variable :")
    missing = df.isnull().sum()
    st.dataframe(missing[missing > 0].to_frame(name="Nombre de manquants"))

with tab2:
    st.subheader("Distributions des variables numériques")
    num_plots = plot_numeric_distributions(df)
    for col_name, fig in num_plots:
        st.pyplot(fig)

    st.subheader("Distributions des variables catégorielles")
    st.pyplot(plot_categorical_distributions(df))

with tab3:
    st.subheader("Matrice de Corrélation")
    st.pyplot(plot_correlation_matrix(df))

    st.subheader("Détection des Valeurs Aberrantes (Méthode IQR)")
    outliers_df = detect_outliers(df)
    st.dataframe(outliers_df, use_container_width=True)

    st.subheader("Boxplots des variables numériques")
    st.pyplot(plot_outliers_boxplots(df))

with tab4:
    st.subheader("Répartition selon le Risque de Santé Mentale")
    st.pyplot(plot_target_relationships(df))

