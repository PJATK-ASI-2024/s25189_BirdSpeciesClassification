from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os

default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def run_script(script_path):
    exit_code = os.system(f"python3 {script_path}")
    if exit_code != 0:
        raise RuntimeError(f"Script {script_path} failed with code {exit_code}")

with DAG(
    dag_id='data_prep_pipeline',
    default_args=default_args,
    description='Data prep pipeline with Google Sheets upload',
    schedule_interval=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
) as dag:

    run_data_prep = PythonOperator(
        task_id='run_data_prep',
        python_callable=run_script,
        op_kwargs={'script_path': '/opt/airflow/scripts/01_data_prep.py'},
    )

    run_eda = PythonOperator(
        task_id='run_eda',
        python_callable=run_script,
        op_kwargs={'script_path': '/opt/airflow/scripts/EDA.py'},
    )

    run_normalization = PythonOperator(
        task_id='run_normalization',
        python_callable=run_script,
        op_kwargs={'script_path': '/opt/airflow/scripts/normalize_metadata.py'},
    )

    upload_to_sheets = PythonOperator(
        task_id='upload_to_sheets',
        python_callable=run_script,
        op_kwargs={'script_path': '/opt/airflow/scripts/upload_to_sheets.py'},
    )

    run_data_prep >> run_eda >> run_normalization >> upload_to_sheets
