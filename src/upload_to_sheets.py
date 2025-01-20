import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd

def push_to_google_sheets(csv_path, sheet_name, credentials_path):
    
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    
    credentials = ServiceAccountCredentials.from_json_keyfile_name(credentials_path, scope)
    client = gspread.authorize(credentials)

    df = pd.read_csv(csv_path)

    try:
        sheet = client.open(sheet_name)
    except gspread.SpreadsheetNotFound:
        sheet = client.create(sheet_name)

    if not sheet.worksheets():
        worksheet = sheet.add_worksheet(title="Sheet1", rows=str(len(df) + 1), cols=str(len(df.columns)))
    else:
        worksheet = sheet.get_worksheet(0)
        
    worksheet.clear()
    worksheet.update([df.columns.values.tolist()] + df.values.tolist())

    print(f"Data successfully uploaded to Google Sheet: {sheet_name}")

if __name__ == "__main__":
    push_to_google_sheets(
        csv_path="/opt/airflow/data/train/normalized_metadata.csv",
        sheet_name="Bird Species Data",
        credentials_path="/opt/airflow/credentials/credentials.json"
    )
