import os
import pandas as pd

# Simulating being in app.py
app_file_path = r"c:\src\dbu.fall.python.final.groupB\Stock_Sentiment_Project\app.py"
daily_csv_path = os.path.join(os.path.dirname(os.path.dirname(app_file_path)), "daily_sentiment_prices.csv")

print(f"Calculated path: {daily_csv_path}")

try:
    new_data = pd.read_csv(daily_csv_path)
    print("Read success.")
    print("Original columns:", new_data.columns.tolist())
    
    new_data = new_data.rename(columns={
        "published_date": "date",
        "avg_close": "close",
        "avg_sentiment": "sentiment_mean"
    })
    
    new_data["date"] = pd.to_datetime(new_data["date"]).dt.strftime("%Y-%m-%d")
    print("Transformation success.")
    print(new_data.head().to_dict(orient="records"))
except Exception as e:
    print(f"Error: {e}")
