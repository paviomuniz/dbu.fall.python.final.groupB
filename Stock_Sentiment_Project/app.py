from flask import Flask, render_template_string, request
import pandas as pd
import pickle
import os

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

ARTIFACT_DIR = "artifacts"

# ------------------------------------------------------------
# Load model, scaler, and data
# ------------------------------------------------------------
with open(os.path.join(ARTIFACT_DIR, "model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(ARTIFACT_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

data = pd.read_csv(os.path.join(ARTIFACT_DIR, "data_with_sentiment.csv"))

# ------------------------------------------------------------
# Normalize column names + auto-detect close/volume/date
# ------------------------------------------------------------
data.columns = [c.lower() for c in data.columns]
print("App data columns:", list(data.columns))

# date column
date_candidates = [c for c in data.columns if "date" in c]
if not date_candidates:
    raise ValueError("Could not find a date column in data_with_sentiment.csv")
date_col = date_candidates[0]

# close price column (e.g. "4. close")
close_candidates = [c for c in data.columns if "close" in c]
if not close_candidates:
    raise ValueError("Could not find a close column in data_with_sentiment.csv")
price_col = close_candidates[0]

# volume column (e.g. "5. volume")
volume_candidates = [c for c in data.columns if "volume" in c]
if not volume_candidates:
    raise ValueError("Could not find a volume column in data_with_sentiment.csv")
volume_col = volume_candidates[0]

# ------------------------------------------------------------
# Prepare sentiment analyzer for free-text headlines
# ------------------------------------------------------------
nltk.download("vader_lexicon", quiet=True)
sia = SentimentIntensityAnalyzer()

# Simple HTML template
TEMPLATE = """
<!doctype html>
<html>
  <head>
    <title>Google Stock & News Sentiment Dashboard</title>
  </head>
  <body>
    <h1>Google Stock & News Sentiment Dashboard</h1>

    <h2>1. Price vs Sentiment (History)</h2>
    <p>Chart generated from your training script:</p>
    <img src="/static/price_sentiment.png" alt="Price vs Sentiment"
         style="max-width: 800px; width: 100%;">

    <h2>2. Last 10 Trading Days</h2>
    <table border="1" cellpadding="4" cellspacing="0">
      <tr>
        <th>Date</th>
        <th>Close</th>
        <th>Average Sentiment</th>
      </tr>
      {% for row in last_rows %}
      <tr>
        <td>{{ row.date }}</td>
        <td>{{ "%.2f"|format(row.close) }}</td>
        <td>{{ "%.3f"|format(row.sentiment_mean) }}</td>
      </tr>
      {% endfor %}
    </table>

    <h2>3. Try Your Own Headline</h2>
    <form method="post">
      <label for="headline">News headline about Google:</label><br>
      <input type="text" id="headline" name="headline" size="80" required>
      <br><br>
      <button type="submit">Predict Price Direction</button>
    </form>

    {% if prediction is not none %}
      <h3>Result</h3>
      <p>Sentiment score: {{ "%.3f"|format(sentiment) }}</p>
      <p>Model prediction for next day price: <strong>{{ prediction }}</strong></p>
    {% endif %}
  </body>
</html>
"""

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction_text = None
    sentiment_score = None

    if request.method == "POST":
        headline = request.form.get("headline", "")
        # Sentiment of user's headline
        sentiment_score = sia.polarity_scores(headline)["compound"]

        # Use last known return and volume as context
        last_row = data.iloc[-1]
        features = pd.DataFrame(
            [{
                "sentiment_mean": sentiment_score,
                "return": last_row["return"],
                "volume": last_row[volume_col],
            }]
        )

        X_scaled = scaler.transform(features.values)
        pred = model.predict(X_scaled)[0]
        prediction_text = "UP 📈" if pred == 1 else "DOWN 📉"

    # Last 10 rows for the table
    last_rows = (
        data[[date_col, price_col, "sentiment_mean"]]
        .tail(10)
        .copy()
    )
    # rename to generic names for the template
    last_rows = last_rows.rename(columns={date_col: "date", price_col: "close"})

    return render_template_string(
        TEMPLATE,
        last_rows=last_rows.to_dict(orient="records"),
        prediction=prediction_text,
        sentiment=sentiment_score,
    )


if __name__ == "__main__":
    app.run(debug=True)
