from pathlib import Path
import sys
import pandas as pd
import streamlit as st

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.data.load_data import load_data
from src.preprocessing.cleaning import split_features_target, split_data
from src.preprocessing.pipeline import build_preprocessor
from src.models.train import get_models, train_and_compare_models
from src.models.predict import load_model
from src.models.evaluate import evaluate_predictions
from streamlit_app.components.visualizations import plot_confusion_matrix_display

st.title("📈 Performances des Modèles")

@st.cache_data
def get_data_splits():
    df = load_data()
    X, y = split_features_target(df)
    return split_data(X, y)

X_train, X_test, y_train, y_test = get_data_splits()

st.subheader("1. Évaluation du Meilleur Modèle")

try:
    model = load_model()
    y_pred = model.predict(X_test)
    classes = model.classes_

    report_dict = evaluate_predictions(y_test, y_pred, output_dict=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{report_dict.get('accuracy', 0):.2%}")
    c2.metric("Precision (Macro)", f"{report_dict.get('macro avg', {}).get('precision', 0):.2%}")
    c3.metric("Recall (Macro)", f"{report_dict.get('macro avg', {}).get('recall', 0):.2%}")
    c4.metric("F1-score (Macro)", f"{report_dict.get('macro avg', {}).get('f1-score', 0):.2%}")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.write("#### Matrice de Confusion")
        fig_cm = plot_confusion_matrix_display(y_test, y_pred, classes)
        st.pyplot(fig_cm)

    with col_right:
        st.write("#### Rapport Détaillé par Classe")
        report_df = pd.DataFrame(report_dict).T
        st.dataframe(report_df.style.format(precision=2), use_container_width=True)

except Exception as e:
    st.warning(f"Le modèle sauvegardé n'a pas pu être chargé ({e}). Entraînez un modèle ci-dessous.")

st.divider()

st.subheader("2. Comparaison des Modèles de Base")
if st.button("🚀 Entraîner et Comparer les Modèles"):
    with st.spinner("Entraînement des modèles en cours..."):
        preprocessor = build_preprocessor(X_train)
        models = get_models()
        results_df, trained_models, best_name, _ = train_and_compare_models(
            models, preprocessor, X_train, y_train, X_test, y_test
        )
        st.session_state["comparison_results"] = results_df
        st.session_state["best_model_name"] = best_name

if "comparison_results" in st.session_state:
    st.write(f"**Meilleur modèle identifié :** `{st.session_state['best_model_name']}`")
    st.dataframe(st.session_state["comparison_results"], use_container_width=True)

    st.bar_chart(
        st.session_state["comparison_results"].set_index("Model")[["Accuracy", "F1-score"]]
    )

