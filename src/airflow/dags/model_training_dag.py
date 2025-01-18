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
    dag_id='model_training_dag',
    default_args=default_args,
    description='Build and train an ML model using processed data',
    schedule_interval=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
) as dag:

    # Task 1: Load & Split data for training
    task_split_data = PythonOperator(
        task_id='split_data_70_30',
        python_callable=run_script,
        op_kwargs={'script_path': '/opt/airflow/scripts/CNN/cnn_prep.py'},
    )

    # Task 2: AutoML or manual training
    task_train_model = PythonOperator(
        task_id='train_model',
        python_callable=run_script,
        op_kwargs={'script_path': '/opt/airflow/scripts/CNN/main.py'},
    )

    # Define the pipeline
    task_split_data >> task_train_model
