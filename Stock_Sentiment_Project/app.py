from flask import Flask, render_template_string, request
import pandas as pd
import pickle
import os
import urllib.request
import urllib.parse
import ssl
import re
import html

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")

# ------------------------------------------------------------
# Load model, scaler, and data (still used for chart/table)
# ------------------------------------------------------------
try:
    with open(os.path.join(ARTIFACT_DIR, "model.pkl"), "rb") as f:
        model = pickle.load(f)
except Exception:
    model = None

try:
    with open(os.path.join(ARTIFACT_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
except Exception:
    scaler = None

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

def extract_text_from_url(url: str) -> str:
    try:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return ""
        ctx = ssl.create_default_context()
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
            },
        )
        with urllib.request.urlopen(req, context=ctx, timeout=10) as resp:
            raw = resp.read()
        try:
            html_text = raw.decode("utf-8", errors="ignore")
        except Exception:
            html_text = raw.decode("latin-1", errors="ignore")
        m = re.search(r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)["\']', html_text, flags=re.I)
        title = m.group(1).strip() if m else ""
        if not title:
            m = re.search(r"<title[^>]*>(.*?)</title>", html_text, flags=re.I | re.S)
            title = re.sub(r"\s+", " ", m.group(1).strip()) if m else ""
        cleaned = re.sub(r"<script[\s\S]*?</script>|<style[\s\S]*?</style>", " ", html_text, flags=re.I)
        cleaned = re.sub(r"<[^>]+>", " ", cleaned)
        cleaned = html.unescape(cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        body_snip = cleaned[:800]
        text = (title + " " + body_snip).strip()
        return text
    except Exception:
        return ""

# ------------------------------------------------------------
# HTML template (same style you had before)
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
    textarea {
      width: 100%;
      padding: 8px 10px;
      border-radius: 8px;
      border: 1px solid #d1d5db;
      font-size: 14px;
      outline: none;
      resize: vertical;
      min-height: 120px;
    }
    textarea:focus {
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
      <h2>3. Paste Content OR Article URL</h2>
      <p class="subtitle">
        Choose one source below. If you enter only a URL, the app will fetch the article and analyze its content. We use VADER to classify it and map to UP / DOWN / UNCERTAIN.
      </p>

      <form method="post">
        <div class="form-group">
          <label>Choose source:</label>
          <label><input type="radio" name="source_type" value="content" {% if selected_source_type == 'content' %}checked{% endif %}> Paste content</label>
          <label style="margin-left:12px;"><input type="radio" name="source_type" value="url" {% if selected_source_type == 'url' %}checked{% endif %}> Article URL</label>
        </div>
        <div class="form-group">
          <label for="content">Paste news content:</label>
          <textarea id="content" name="content" placeholder="Paste the article text here..."></textarea>
        </div>
        <div class="form-group">
          <label for="url">News article URL (optional):</label>
          <input type="text" id="url" name="url" placeholder="https://example.com/news/google-article">
        </div>
        <button type="submit">Predict Price Direction</button>
      </form>

      <script>
        (function(){
          function toggle(){
            var source = document.querySelector('input[name="source_type"]:checked');
            var isContent = !source || source.value === 'content';
            var contentEl = document.getElementById('content');
            var urlEl = document.getElementById('url');
            var contentGroup = contentEl.parentElement;
            var urlGroup = urlEl.parentElement;
            contentGroup.style.display = isContent ? 'block' : 'none';
            urlGroup.style.display = isContent ? 'none' : 'block';
            contentEl.required = isContent;
            urlEl.required = !isContent;
          }
          var radios = document.querySelectorAll('input[name="source_type"]');
          radios.forEach(function(r){ r.addEventListener('change', toggle); });
          toggle();
        })();
      </script>

      {% if prediction is not none %}
        <div class="prediction {{ css_class }}">
          <span class="label">Prediction:</span> {{ prediction }}
        </div>
        <div class="sentiment-score">
          Sentiment (VADER compound): {{ "%.3f"|format(sentiment) }}
        </div>
        {% if analyzed_text %}
        <div class="sentiment-score">
          Analyzed source: {{ analyzed_source }}
        </div>
        <div class="sentiment-score">
          Text snippet: {{ analyzed_text|truncate(300) }}
        </div>
        {% endif %}
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
    analyzed_text = ""
    analyzed_source = ""

    if request.method == "POST":
        source_type = request.form.get("source_type", "content").strip()
        content = request.form.get("content", "").strip()
        url = request.form.get("url", "").strip()
        text_to_analyze = ""
        if source_type == "content":
            if content:
                text_to_analyze = content
                analyzed_text = content
                analyzed_source = "Content"
            elif url:
                text_to_analyze = extract_text_from_url(url)
                if text_to_analyze:
                    analyzed_text = text_to_analyze
                    analyzed_source = "URL"
        else:  # source_type == "url"
            if url:
                text_to_analyze = extract_text_from_url(url)
                if text_to_analyze:
                    analyzed_text = text_to_analyze
                    analyzed_source = "URL"
            elif content:
                text_to_analyze = content
                analyzed_text = content
                analyzed_source = "Content"
        if text_to_analyze:
            sentiment_score = sia.polarity_scores(text_to_analyze)["compound"]

        # SIMPLE, EXPLAINABLE RULE:
        # sentiment >= 0.05   -> UP
        # sentiment <= -0.05  -> DOWN
        # otherwise           -> UNCERTAIN
        if sentiment_score is not None:
            if sentiment_score >= 0.05:
                prediction_text = "UP 📈"
                css_class = "up"
            elif sentiment_score <= -0.05:
                prediction_text = "DOWN 📉"
                css_class = "down"
            else:
                prediction_text = "UNCERTAIN 🤔"
                css_class = "neutral"

    # show actual last 10 rows (even if sentiment is 0 = no news)
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
        analyzed_text=analyzed_text,
        analyzed_source=analyzed_source,
        selected_source_type=request.form.get("source_type", "content") if request.method == "POST" else "content",
    )


if __name__ == "__main__":
    app.run(debug=True)
