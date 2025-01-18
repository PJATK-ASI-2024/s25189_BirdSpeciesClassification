from airflow import DAG
from airflow.operators.email import EmailOperator
from airflow.utils.dates import days_ago
from datetime import datetime

default_args = {
    'owner': 'airflow'
}

with DAG(
    dag_id='test_email_dag',
    default_args=default_args,
    start_date=days_ago(1),
    schedule_interval=None,
    catchup=False
) as dag:

    send_test_email = EmailOperator(
        task_id='send_test_email',
        to='somebody@example.com',  # Could be your own email
        subject='Airflow Test Email',
        html_content='<h3>Hello from Airflow!</h3>'
    )