
FROM apache/airflow:2.9.2-python3.11

USER airflow

RUN pip install --no-cache-dir \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.9.2/constraints-3.11.txt" \
    pandas \
    numpy \
    scikit-learn \
    seaborn \
    matplotlib \
    pyyaml \
    joblib
