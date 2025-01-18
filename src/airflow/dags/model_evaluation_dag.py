import os
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

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
    dag_id='model_evaluation_dag',
    default_args=default_args,
    description='DAG that evaluates a trained model',
    schedule_interval=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
) as dag:

    evaluate_model_task = PythonOperator(
        task_id='evaluate_model',
        python_callable=run_script,
        op_kwargs={'script_path': '/opt/airflow/scripts/evaluate_model.py'},
    )
