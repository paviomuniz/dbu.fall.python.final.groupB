from flask import Flask, render_template_string, request
import pandas as pd
import pickle
import os
from datetime import datetime

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

ARTIFACT_DIR = "artifacts"

# ------------------------------------------------------------
# Load model, scaler, and data (still used for chart/table)
# ------------------------------------------------------------
with open(os.path.join(ARTIFACT_DIR, "model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(ARTIFACT_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

data = pd.read_csv(os.path.join(ARTIFACT_DIR, "data_with_sentiment.csv"))
data.columns = [c.lower() for c in data.columns]

# detect key columns
date_col = [c for c in data.columns if "date" in c][0]
price_col = [c for c in data.columns if "close" in c][0]
volume_col = [c for c in data.columns if "volume" in c][0]

# ------------------------------------------------------------
# Sentiment analyzer for custom headlines
# ------------------------------------------------------------
nltk.download("vader_lexicon", quiet=True)
sia = SentimentIntensityAnalyzer()

# Simple keyword lists to "help" VADER on very short headlines
POSITIVE_HINTS = [
    "making money",
    "doing great",
    "great in cloud",
    "record-breaking",
    "record breaking",
    "strong growth",
    "strong results",
    "profit",
    "profits",
    "growing",
    "better than expected",
    "beats expectations",
    "beat expectations",
]

NEGATIVE_HINTS = [
    "not doing great",
    "not good",
    "losing money",
    "losing",
    "loss",
    "went down",
    "stocks went down",
    "is down",
    "down in cloud",
    "bad",
    "lawsuit",
    "antitrust",
    "fine",
    "regulatory action",
    "drop",
    "dropped",
    "fell",
    "falling",
]

# ------------------------------------------------------------
# Simple in-memory history of last 10 predictions
# ------------------------------------------------------------
HEADLINE_HISTORY = []  # each item: dict(timestamp, headline, sentiment, prediction)

# ------------------------------------------------------------
# HTML template (with history + reset button)
# ------------------------------------------------------------
TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Google Stock & News Sentiment Dashboard</title>
  <style>
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f3f4f8;
      color: #111827;
    }
    .header {
      background: linear-gradient(120deg, #2563eb, #10b981);
      padding: 20px 30px;
      color: white;
      box-shadow: 0 2px 8px rgba(15, 23, 42, 0.3);
    }
    .header h1 { margin: 0; font-size: 26px; font-weight: 600; }
    .header p { margin: 6px 0 0; font-size: 14px; opacity: 0.9; }

    .page {
      max-width: 1100px;
      margin: 25px auto 40px;
      padding: 0 16px;
    }
    .grid {
      display: grid;
      grid-template-columns: minmax(0, 2fr) minmax(0, 1.5fr);
      gap: 20px;
      margin-bottom: 20px;
    }
    .card {
      background: white;
      border-radius: 14px;
      padding: 18px 20px;
      box-shadow: 0 1px 6px rgba(15, 23, 42, 0.12);
      margin-bottom: 18px;
    }
    .card h2 {
      margin: 0 0 10px;
      font-size: 18px;
      border-bottom: 1px solid #e5e7eb;
      padding-bottom: 6px;
    }
    .card p.subtitle {
      margin: 0 0 14px;
      font-size: 13px;
      color: #6b7280;
    }
    img.chart {
      max-width: 100%;
      border-radius: 10px;
      border: 1px solid #e5e7eb;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    th, td {
      padding: 6px 8px;
      text-align: left;
      border-bottom: 1px solid #e5e7eb;
      vertical-align: top;
    }
    th { background: #f9fafb; font-weight: 600; }
    tr:nth-child(even) td { background: #f9fafb; }

    .form-group { margin-bottom: 10px; }
    label { display: block; font-size: 14px; margin-bottom: 4px; }
    input[type="text"] {
      width: 100%;
      padding: 8px 10px;
      border-radius: 8px;
      border: 1px solid #d1d5db;
      font-size: 14px;
      outline: none;
    }
    input[type="text"]:focus {
      border-color: #2563eb;
      box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.15);
    }
    button {
      padding: 8px 14px;
      border-radius: 999px;
      border: none;
      cursor: pointer;
      background: #2563eb;
      color: white;
      font-size: 14px;
      font-weight: 500;
      margin-top: 4px;
    }
    button:hover { background: #1d4ed8; }

    .btn-reset {
      background: #6b7280;
      margin-bottom: 8px;
    }
    .btn-reset:hover {
      background: #4b5563;
    }

    .prediction {
      margin-top: 12px;
      padding: 10px 12px;
      border-radius: 10px;
      font-size: 14px;
      display: inline-block;
    }
    .prediction.up {
      background: #ecfdf3;
      color: #166534;
      border: 1px solid #bbf7d0;
    }
    .prediction.down {
      background: #fef2f2;
      color: #b91c1c;
      border: 1px solid #fecaca;
    }
    .prediction.neutral {
      background: #eff6ff;
      color: #1d4ed8;
      border: 1px solid #bfdbfe;
    }
    .prediction .label { font-weight: 600; margin-right: 4px; }
    .sentiment-score {
      font-size: 13px;
      color: #4b5563;
      margin-top: 4px;
    }
    .footer {
      margin-top: 24px;
      font-size: 12px;
      color: #9ca3af;
      text-align: center;
    }
  </style>
</head>
<body>
  <div class="header">
    <h1>Google Stock & News Sentiment Dashboard</h1>
    <p>Trained ML model using daily stock prices + news headline sentiment.</p>
  </div>

  <div class="page">
    <div class="grid">
      <!-- Chart card -->
      <div class="card">
        <h2>1. Price vs Sentiment (History)</h2>
        <p class="subtitle">Historical closing price and average daily news sentiment.</p>
        <img src="/static/price_sentiment.png" alt="Price vs Sentiment" class="chart">
      </div>

      <!-- Last 10 trading days -->
      <div class="card">
        <h2>2. Last 10 Trading Days</h2>
        <p class="subtitle">Recent closing prices and average sentiment (0 means no news that day).</p>
        <table>
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
      </div>
    </div>

    <!-- Prediction form -->
    <div class="card">
      <h2>3. Try Your Own Headline</h2>
      <p class="subtitle">
        Type a news headline about Google. We use VADER + simple keyword rules to classify it
        as positive, negative, or neutral and then map that to UP / DOWN / UNCERTAIN.
      </p>

      <form method="post">
        <input type="hidden" name="action" value="predict">
        <div class="form-group">
          <label for="headline">News headline about Google:</label>
          <input type="text" id="headline" name="headline"
                 placeholder="Example: Google announces record-breaking quarterly earnings..." required>
        </div>
        <button type="submit">Predict Price Direction</button>
      </form>

      {% if prediction is not none %}
        <div class="prediction {{ css_class }}">
          <span class="label">Prediction:</span> {{ prediction }}
        </div>
        <div class="sentiment-score">
          Headline sentiment (final score): {{ "%.3f"|format(sentiment) }}
        </div>
      {% endif %}
    </div>

    <!-- History card -->
    <div class="card">
      <h2>4. Recent Headline Predictions</h2>
      <p class="subtitle">Last 10 headlines you tested, with their sentiment score and predicted direction.</p>

      <form method="post">
        <input type="hidden" name="action" value="reset_history">
        <button type="submit" class="btn-reset">Reset history</button>
      </form>

      {% if history %}
      <table>
        <tr>
          <th>Time</th>
          <th>Headline</th>
          <th>Sentiment</th>
          <th>Prediction</th>
        </tr>
        {% for item in history %}
        <tr>
          <td style="white-space: nowrap;">{{ item.timestamp }}</td>
          <td>{{ item.headline }}</td>
          <td>{{ "%.3f"|format(item.sentiment) }}</td>
          <td>{{ item.prediction }}</td>
        </tr>
        {% endfor %}
      </table>
      {% else %}
      <p class="subtitle">No headlines tested yet. Try entering one above.</p>
      {% endif %}
    </div>

    <div class="footer">
      Academic project • This interactive demo uses a simple sentiment-based rule
      (not a real trading system).
    </div>
  </div>
</body>
</html>
"""

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction_text = None
    sentiment_score = None
    css_class = ""

    if request.method == "POST":
        action = request.form.get("action", "predict")

        if action == "reset_history":
            # Clear the stored headlines
            HEADLINE_HISTORY.clear()

        elif action == "predict":
            headline = request.form.get("headline", "")
            base_score = sia.polarity_scores(headline)["compound"]

            # Apply simple keyword boost to help VADER on very short texts
            text_lower = headline.lower()
            boost = 0.0

            for kw in POSITIVE_HINTS:
                if kw in text_lower:
                    boost += 0.4
                    break

            for kw in NEGATIVE_HINTS:
                if kw in text_lower:
                    boost -= 0.4
                    break

            # Final sentiment score, clipped to [-1, 1]
            sentiment_score = max(-1.0, min(1.0, base_score + boost))

            # More sensitive thresholds: fewer "UNCERTAIN"
            if sentiment_score >= 0.02:
                prediction_text = "UP 📈"
                css_class = "up"
            elif sentiment_score <= -0.02:
                prediction_text = "DOWN 📉"
                css_class = "down"
            else:
                prediction_text = "UNCERTAIN 🤔"
                css_class = "neutral"

            # Store in history (max 10 items)
            HEADLINE_HISTORY.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "headline": headline,
                "sentiment": sentiment_score,
                "prediction": prediction_text,
            })
            if len(HEADLINE_HISTORY) > 10:
                HEADLINE_HISTORY.pop(0)  # remove oldest

    # show last 10 calendar days (0 sentiment = no news)
    last_rows = (
        data[[date_col, price_col, "sentiment_mean"]]
        .tail(10)
        .copy()
    )
    last_rows = last_rows.rename(columns={date_col: "date", price_col: "close"})

    return render_template_string(
        TEMPLATE,
        last_rows=last_rows.to_dict(orient="records"),
        prediction=prediction_text,
        sentiment=sentiment_score,
        css_class=css_class,
        history=HEADLINE_HISTORY,
    )


if __name__ == "__main__":
    app.run(debug=True)
