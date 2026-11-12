from pathlib import Path
import sys
import pandas as pd
import streamlit as st

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.models.predict import load_model, predict, predict_proba

st.title("🔮 Test du Modèle & Prédiction en Direct")

try:
    model = load_model()
except Exception as e:
    st.error(f"Impossible de charger le modèle : {e}")
    st.stop()

tab_single, tab_batch = st.tabs(["Prédiction Individuelle", "Prédiction par Lot (CSV)"])

with tab_single:
    st.markdown("Renseignez les caractéristiques de l'étudiant pour évaluer son niveau de risque :")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### 👤 Profil Académique")
            age = st.number_input("Âge", min_value=16, max_value=40, value=21)
            gender = st.selectbox("Genre", ["Male", "Female", "Other"])
            year_of_study = st.selectbox("Année d'étude", [1, 2, 3, 4, 5], index=1)
            gpa = st.slider("Moyenne Académique (GPA)", min_value=0.0, max_value=4.0, value=3.2, step=0.05)
            attendance = st.slider("Taux de Présence (%)", min_value=0.0, max_value=100.0, value=85.0, step=1.0)

        with col2:
            st.markdown("##### ⏱️ Rythme de Vie")
            sleep_hours = st.slider("Heures de Sommeil / Jour", min_value=0.0, max_value=14.0, value=7.0, step=0.5)
            study_hours = st.slider("Heures d'Étude / Semaine", min_value=0.0, max_value=80.0, value=20.0, step=1.0)
            social_media = st.slider("Réseaux Sociaux (h/jour)", min_value=0.0, max_value=16.0, value=3.0, step=0.5)
            screen_time = st.slider("Temps d'Écran Total (h/jour)", min_value=0.0, max_value=24.0, value=6.0, step=0.5)

        with col3:
            st.markdown("##### 🧘 Facteurs Personnels & Santé")
            physical_activity = st.selectbox("Activité Physique", ["Low", "Moderate", "High"])
            diet_quality = st.selectbox("Qualité Alimentaire", ["Poor", "Average", "Good"])
            financial_stress = st.slider("Stress Financier (1 à 5)", min_value=1.0, max_value=5.0, value=3.0, step=1.0)
            family_support = st.slider("Soutien Familial (1 à 5)", min_value=1.0, max_value=5.0, value=4.0, step=1.0)
            part_time_job = st.selectbox("Emploi à Temps Partiel", ["No", "Yes"])
            previous_history = st.selectbox("Antécédents de Santé Mentale", ["No", "Yes"])

        submitted = st.form_submit_button("🔍 Prédire le Niveau de Risque")

    if submitted:
        input_data = pd.DataFrame([{
            "Student_ID": "STU_TEMP",
            "Age": age,
            "Gender": gender,
            "Year_of_Study": year_of_study,
            "Sleep_Hours": sleep_hours,
            "Study_Hours_Per_Week": study_hours,
            "GPA": gpa,
            "Social_Media_Hours": social_media,
            "Screen_Time_Hours": screen_time,
            "Physical_Activity": physical_activity,
            "Financial_Stress": financial_stress,
            "Family_Support": family_support,
            "Attendance_Rate": attendance,
            "Part_Time_Job": part_time_job,
            "Diet_Quality": diet_quality,
            "Previous_Mental_Health_History": previous_history,
        }])

        prediction = predict(model, input_data)[0]
        color_badges = {
            "Low": "🟢 Risque Faible (Low)",
            "Medium": "🟡 Risque Modéré (Medium)",
            "High": "🔴 Risque Élevé (High)",
        }

        st.subheader("Résultat :")
        if prediction == "High":
            st.error(f"### {color_badges.get(prediction, prediction)}")
            st.warning("Recommandation : Proposer un accompagnement psychologique et un suivi pédagogique adapté.")
        elif prediction == "Medium":
            st.warning(f"### {color_badges.get(prediction, prediction)}")
            st.info("Recommandation : Surveiller le stress académique et encourager un meilleur équilibre sommeil/écran.")
        else:
            st.success(f"### {color_badges.get(prediction, prediction)}")
            st.info("Recommandation : Indicateurs équilibrés. Poursuivre le suivi de routine.")

        probas = predict_proba(model, input_data)
        if probas is not None:
            st.write("##### Probabilités calculées par classe :")
            proba_df = pd.DataFrame(probas, columns=model.classes_)
            st.dataframe(proba_df.style.format("{:.2%}"), use_container_width=True)

with tab_batch:
    st.markdown("Importez un fichier CSV contenant les mêmes colonnes pour obtenir les prédictions en masse :")
    uploaded_file = st.file_uploader("Fichier CSV à prédire", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write("Aperçu des données importées :", batch_df.head(5))

        if "Student_ID" not in batch_df.columns:
            batch_df["Student_ID"] = [f"STU_{i+1:04d}" for i in range(len(batch_df))]

        features_df = batch_df.drop(columns=["Mental_Health_Risk"], errors="ignore")
        batch_preds = predict(model, features_df)

        batch_result = batch_df.copy()
        batch_result["Predicted_Risk"] = batch_preds

        st.success(f"Prédictions effectuées pour {len(batch_result)} étudiants !")
        st.dataframe(batch_result[["Student_ID", "Predicted_Risk"] + [c for c in batch_result.columns if c not in ["Student_ID", "Predicted_Risk"]]].head(15), use_container_width=True)

        csv_download = batch_result.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Télécharger les résultats (CSV)",
            data=csv_download,
            file_name="predictions_mental_health_risk.csv",
            mime="text/csv",
        )

