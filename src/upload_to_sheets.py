import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd

def push_to_google_sheets(csv_path, sheet_name, credentials_path):
    """
    Push data from a CSV file to a Google Sheet.

    Args:
        csv_path (str): Path to the CSV file.
        sheet_name (str): Name of the Google Sheet.
        credentials_path (str): Path to the service account credentials file.
    """
    # Define the Google Sheets API scope
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    
    # Authenticate using the service account credentials
    credentials = ServiceAccountCredentials.from_json_keyfile_name(credentials_path, scope)
    client = gspread.authorize(credentials)

    # Read the data from the CSV file
    df = pd.read_csv(csv_path)

    # Open the Google Sheet or create a new one
    try:
        sheet = client.open(sheet_name)
    except gspread.SpreadsheetNotFound:
        sheet = client.create(sheet_name)

    # Select the first worksheet, or create it if it doesn't exist
    if not sheet.worksheets():
        worksheet = sheet.add_worksheet(title="Sheet1", rows=str(len(df) + 1), cols=str(len(df.columns)))
    else:
        worksheet = sheet.get_worksheet(0)

    # Clear the worksheet and update with new data
    worksheet.clear()
    worksheet.update([df.columns.values.tolist()] + df.values.tolist())

    print(f"Data successfully uploaded to Google Sheet: {sheet_name}")

# Example usage
if __name__ == "__main__":
    push_to_google_sheets(
        csv_path="/opt/airflow/data/train/normalized_metadata.csv",
        sheet_name="Bird Species Data",
        credentials_path="/opt/airflow/credentials/credentials.json"
    )
