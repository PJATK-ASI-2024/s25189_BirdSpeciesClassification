import os
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report

TEST_DATA_PATH = "/opt/airflow/data/train/test_ml.csv"
MODEL_PATH = "/opt/airflow/models/best_model.pkl"
REPORT_PATH = "/opt/airflow/reports/evaluation_report.txt"

def main():
    # 1. Load test data
    df = pd.read_csv(TEST_DATA_PATH)
    # Adjust column names to match your project if needed
    y_true = df['class_id']  # or whatever column is your target
    X_test = df.drop(['class_id','image_id','image_path','classes_name'], axis=1, errors='ignore')

    model = joblib.load(MODEL_PATH)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)

    print("Accuracy:", acc)
    print(report)

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write(f"Accuracy: {acc}\n\n")
        f.write(report)

    print(f"Evaluation report saved to: {REPORT_PATH}")

if __name__ == "__main__":
    main()
