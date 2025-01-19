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

# DAG 1: Data Preparation and Upload
data_prep_upload_dag = DAG(
    dag_id='data_prep_upload_pipeline',
    default_args=default_args,
    description='Pipeline for data preparation and Google Sheets upload',
    schedule_interval=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
)

with data_prep_upload_dag:
    run_data_prep = PythonOperator(
        task_id='run_data_prep',
        python_callable=run_script,
        op_kwargs={'script_path': '/opt/airflow/scripts/01_data_prep.py'},
    )

    upload_to_sheets = PythonOperator(
        task_id='upload_to_sheets',
        python_callable=run_script,
        op_kwargs={'script_path': '/opt/airflow/scripts/upload_to_sheets.py'},
    )

    run_data_prep >> upload_to_sheets

# DAG 2: EDA and Normalization
eda_normalization_dag = DAG(
    dag_id='eda_normalization_pipeline',
    default_args=default_args,
    description='Pipeline for EDA and metadata normalization',
    schedule_interval=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
)

with eda_normalization_dag:
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

    run_eda >> run_normalization