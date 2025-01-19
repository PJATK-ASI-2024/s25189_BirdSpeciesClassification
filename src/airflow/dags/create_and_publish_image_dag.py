from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from airflow.models import Variable

# Define default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
}

# Define the DAG
with DAG(
    dag_id='containerize_and_publish_image',
    default_args=default_args,
    description='Containerize API and publish Docker image to Docker Hub',
    schedule_interval=None,  # Trigger manually
    start_date=days_ago(1),
    catchup=False,
    tags=['docker', 'api'],
) as dag:

    # Step 1: Build Docker Image
    build_docker_image = BashOperator(
        task_id='build_docker_image',
        bash_command=(
            "docker build -t kneiv/model-and-api-bird-classifier:latest "
            "-f /opt/airflow/backend_container/Dockerfile /opt/airflow/backend_container"
        ),
    )

    # Step 2: Login to Docker Hub
    login_to_docker_hub = BashOperator(
        task_id='login_to_docker_hub',
        bash_command=(
            "echo $DOCKER_HUB_PASSWORD | docker login --username $DOCKER_HUB_USERNAME --password-stdin"
        ),
        env={
            'DOCKER_HUB_USERNAME': Variable.get("DOCKER_HUB_USERNAME"),
            'DOCKER_HUB_PASSWORD': Variable.get("DOCKER_HUB_PASSWORD"),
        },
    )

    # Step 3: Push Docker Image to Docker Hub
    push_docker_image = BashOperator(
        task_id='push_docker_image',
        bash_command=(
            "docker push kneiv/model-and-api-bird-classifier:latest"
        ),
    )

    # Define task dependencies
    build_docker_image >> login_to_docker_hub >> push_docker_image
