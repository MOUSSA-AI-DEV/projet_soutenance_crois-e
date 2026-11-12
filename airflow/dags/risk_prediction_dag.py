from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator


def prepare_data():
    from src.data.load_data import load_data
    from src.data.validate_data import validate_data

    df = load_data()
    validate_data(df)


def preprocess_data():
    from src.data.load_data import load_data
    from src.preprocessing.cleaning import split_features_target, split_data
    from src.preprocessing.outliers import detect_outliers

    df = load_data()
    detect_outliers(df)
    split_features_target(df)


def train_model():
    from src.data.load_data import load_data
    from src.preprocessing.cleaning import split_features_target, split_data
    from src.preprocessing.pipeline import build_preprocessor
    from src.models.train import get_models, train_and_compare_models
    from src.models.predict import save_model

    df = load_data()
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    preprocessor = build_preprocessor(X_train)
    models = get_models()
    _, _, _, best_pipeline = train_and_compare_models(
        models, preprocessor, X_train, y_train, X_test, y_test
    )
    save_model(best_pipeline)


def evaluate_model():
    from src.data.load_data import load_data
    from src.preprocessing.cleaning import split_features_target, split_data
    from src.models.predict import load_model, predict
    from src.models.evaluate import evaluate_predictions, save_metrics

    df = load_data()
    X, y = split_features_target(df)
    _, X_test, _, y_test = split_data(X, y)

    model = load_model()
    y_pred = predict(model, X_test)
    report = evaluate_predictions(y_test, y_pred, output_dict=True)
    save_metrics(report)


with DAG(
    "mental_health_risk_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule_interval="0 2 * * *",
    catchup=False,
) as dag:

    prepare_task = PythonOperator(
        task_id="preparation_donnees",
        python_callable=prepare_data,
    )

    preprocess_task = PythonOperator(
        task_id="pretraitement_donnees",
        python_callable=preprocess_data,
    )

    train_task = PythonOperator(
        task_id="entrainement_modele",
        python_callable=train_model,
    )

    evaluate_task = PythonOperator(
        task_id="evaluation_modele",
        python_callable=evaluate_model,
    )

    prepare_task >> preprocess_task >> train_task >> evaluate_task
