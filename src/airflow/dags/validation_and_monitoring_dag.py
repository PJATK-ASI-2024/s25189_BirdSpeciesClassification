import os
import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.email import EmailOperator
from airflow.operators.dummy import DummyOperator

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='model_validation_monitoring',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    description='Validate model on images, run pipeline unit tests, send expanded email alerts with test logs.'
) as dag:

    # ------------------------------------------------------------------------------
    # 1. CONFIG
    MODEL_NAME = "BirdClassifier"
    THRESHOLD = 0.80
    MODEL_PATH = "/opt/airflow/saved_models/best_model.pth"
    VALIDATION_CSV = "/opt/airflow/data/val/metadata.csv"
    IMAGE_ROOT = "/opt/airflow/data/val/images"
    TEST_SCRIPT = "/opt/airflow/scripts/CNN/test_trainer.py"
    NUM_CLASSES = 200
    BATCH_SIZE = 16

    # ------------------------------------------------------------------------------
    # 2. A custom Dataset for validation
    class BirdValDataset(Dataset):
        def __init__(self, csv_path, image_root, transform=None):
            self.df = pd.read_csv(csv_path)
            self.image_root = image_root
            self.transform = transform

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            image_rel = row['image_path']
            label = row['class_id']

            img_full = os.path.join(self.image_root, image_rel)
            img = Image.open(img_full).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, label

    # ------------------------------------------------------------------------------
    # 3. Task: Load model, run inference => accuracy
    def load_and_evaluate(**context):
        logging.info("Loading .pth model & evaluating on validation images...")

        # a) define model arch
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(2048, NUM_CLASSES)

        # b) load state_dict
        sd = torch.load(MODEL_PATH, map_location='cpu')
        # if checkpoint has 'model.' prefix, remove it
        new_sd = {}
        for k, v in sd.items():
            if k.startswith("model."):
                new_k = k.replace("model.", "")
                new_sd[new_k] = v
            else:
                new_sd[k] = v

        missing, unexpected = model.load_state_dict(new_sd, strict=False)
        logging.info(f"Missing keys: {missing}")
        logging.info(f"Unexpected keys: {unexpected}")
        model.eval()

        # c) data transforms
        val_transform = T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
        ])

        # d) create dataset / loader
        ds = BirdValDataset(csv_path=VALIDATION_CSV, image_root=IMAGE_ROOT, transform=val_transform)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

        # e) gather predictions
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in loader:
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_labels)

        acc = accuracy_score(y_true, y_pred)
        logging.info(f"Validation Accuracy: {acc:.4f}")

        # store in XCom
        context['ti'].xcom_push(key='model_accuracy', value=acc)

    compute_metrics_task = PythonOperator(
        task_id='compute_metrics',
        python_callable=load_and_evaluate,
        provide_context=True
    )

    # ------------------------------------------------------------------------------
    # 4. Task: run unit tests
    #    Capture stdout in XCom to embed in email
    def run_tests(**context):
        import subprocess

        proc = subprocess.run(["python3", TEST_SCRIPT],
                              capture_output=True, text=True)
        exit_code = proc.returncode
        output = proc.stdout + "\n" + proc.stderr

        if exit_code == 0:
            logging.info("Tests passed!")
            test_status = "passed"
        else:
            logging.info("Tests failed!")
            test_status = "failed"

        # store both status & logs in XCom
        context['ti'].xcom_push(key='test_status', value=test_status)
        context['ti'].xcom_push(key='test_logs', value=output)

    run_tests_task = PythonOperator(
        task_id='run_tests',
        python_callable=run_tests,
        provide_context=True
    )

    # ------------------------------------------------------------------------------
    # 5. Branch to decide if we email
    def check_results(**context):
        acc = context['ti'].xcom_pull(key='model_accuracy', task_ids='compute_metrics')
        test_status = context['ti'].xcom_pull(key='test_status', task_ids='run_tests')

        if acc is None:
            acc = 0.0

        if acc < THRESHOLD or test_status == "failed":
            logging.info(f"Triggered alert: acc={acc}, test_status={test_status}")
            return 'send_email_alert'
        else:
            logging.info(f"No alert: acc={acc}, test_status={test_status}")
            return 'no_alert'

    decide_task = BranchPythonOperator(
        task_id='decide_alert',
        python_callable=check_results,
        provide_context=True
    )

    no_alert_task = DummyOperator(task_id='no_alert')

    # ------------------------------------------------------------------------------
    # 6. Email alert => expanded content
    #    We embed test logs if the tests failed
    def build_email_content(**context):
        """
        If tests failed, embed test logs in the email body.
        Also show the accuracy.
        """
        ti = context['ti']
        acc = ti.xcom_pull(key='model_accuracy', task_ids='compute_metrics')
        test_status = ti.xcom_pull(key='test_status', task_ids='run_tests')
        test_logs = ti.xcom_pull(key='test_logs', task_ids='run_tests')

        if acc is None:
            acc = 0.0
        if test_logs is None:
            test_logs = "No test logs available."

        # We can build an HTML string
        html_content = f"""
            <h3>Model Validation Alert</h3>
            <p>Model: {MODEL_NAME}</p>
            <p>Accuracy: {acc:.4f} (Threshold = {THRESHOLD})</p>
            <p>Test Status: {test_status}</p>
        """

        # If tests failed, show logs
        if test_status == 'failed':
            # replace newlines with <br> for HTML
            logs_html = test_logs.replace('\n', '<br>')
            html_content += f"<h4>Test Logs (failure):</h4><pre>{logs_html}</pre>"
        else:
            html_content += "<p>Tests passed successfully.</p>"

        return html_content

    def send_email(**context):
        from airflow.utils.email import send_email
        ti = context['ti']
        html_body = ti.xcom_pull(key='email_body', task_ids='build_email_body')

        # "send_email" is a built-in function in airflow
        send_email(
            to=['team@example.com'],
            subject='[Airflow Alert] Model Validation Issue',
            html_content=html_body
        )

    build_email_body = PythonOperator(
        task_id='build_email_body',
        python_callable=build_email_content,
        provide_context=True
    )

    send_email_op = PythonOperator(
        task_id='send_email_alert',
        python_callable=send_email,
        provide_context=True
    )

    # We only call build_email_body + send_email_op if branched
    # So we chain them
    build_email_body >> send_email_op

    # ------------------------------------------------------------------------------
    # DAG pipeline
    [compute_metrics_task, run_tests_task] >> decide_task
    decide_task >> no_alert_task
    decide_task >> build_email_body
