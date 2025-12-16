import os
import pickle
import pandas as pd
import numpy as np
 

from flask import Blueprint, request, render_template_string

from typing import Dict

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer


 

def run_training(stock_path: str, news_path: str) -> Dict[str, float]:
    """Run end-to-end training using the given local CSV paths.

    Loads stock and news CSVs, computes VADER sentiment, merges daily
    averages with stock data, trains Logistic Regression and Random Forest,
    saves artifacts to `artifacts/` and chart to `static/price_sentiment.png`.
    Returns a dict with summary metrics for display."""
    stocks = pd.read_csv(stock_path)
    news = pd.read_csv(news_path)

    stocks.columns = [c.lower() for c in stocks.columns]
    news.columns = [c.lower() for c in news.columns]

    stock_date_candidates = [c for c in stocks.columns if "date" in c]
    if not stock_date_candidates:
        raise ValueError("Could not find a date column in the stock CSV.")
    date_col_stock = stock_date_candidates[0]

    close_candidates = [c for c in stocks.columns if "close" in c]
    if not close_candidates:
        raise ValueError("Could not find a close/adj close column in the stock CSV.")
    close_col = close_candidates[0]

    volume_candidates = [c for c in stocks.columns if "volume" in c]
    if not volume_candidates:
        raise ValueError("Could not find a volume column in the stock CSV.")
    volume_col = volume_candidates[0]

    news_date_candidates = [
        c for c in news.columns
        if "date" in c or "time" in c or "publish" in c
    ]
    if not news_date_candidates:
        raise ValueError("Could not find a date/published column in the news CSV.")
    date_col_news = news_date_candidates[0]

    news_title_candidates = [
        c for c in news.columns
        if "title" in c or "headline" in c or "news" in c
    ]
    if not news_title_candidates:
        raise ValueError("Could not find a title/headline column in the news CSV.")
    title_col = news_title_candidates[0]

    stocks[date_col_stock] = pd.to_datetime(stocks[date_col_stock])
          
    
    stocks = stocks.sort_values(date_col_stock)

    news[date_col_news] = pd.to_datetime(news[date_col_news])

    stocks["return"] = stocks[close_col].pct_change()
    stocks["target"] = (stocks[close_col].shift(-1) > stocks[close_col]).astype(int)
    stocks = stocks.dropna(subset=["return", "target"])

    nltk.download("vader_lexicon", quiet=True)
    sia = SentimentIntensityAnalyzer()
    news["sentiment"] = news[title_col].astype(str).apply(
        lambda text: sia.polarity_scores(text)["compound"]
    )

    news_daily = (
        news.groupby(news[date_col_news].dt.date)["sentiment"]
        .mean()
        .reset_index()
        .rename(columns={date_col_news: "date_only", "sentiment": "sentiment_mean"})
    )

    stocks["date_only"] = stocks[date_col_stock].dt.date
    merged = pd.merge(
        stocks, news_daily, how="left", left_on="date_only", right_on="date_only"
    )
    merged["sentiment_mean"] = merged["sentiment_mean"].fillna(0.0)

    feature_cols = ["sentiment_mean", "return", volume_col]
    X = merged[feature_cols].values
    y = merged["target"].values
    mask = ~np.isnan(X).any(axis=1)
    X = X[mask]
    y = y[mask]
    merged = merged.loc[mask].reset_index(drop=True)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    from sklearn.ensemble import RandomForestClassifier
    rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf_model.fit(X_train, y_train)

    from sklearn.metrics import accuracy_score
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    rf_pred = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)

    os.makedirs("artifacts", exist_ok=True)
    with open(os.path.join("artifacts", "model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join("artifacts", "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    merged.to_csv(os.path.join("artifacts", "data_with_sentiment.csv"), index=False)

    import matplotlib.pyplot as plt
    os.makedirs("static", exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(merged[date_col_stock], merged[close_col], label="Close Price")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Close Price")

    ax2 = ax1.twinx()
    ax2.plot(merged[date_col_stock], merged["sentiment_mean"], label="Sentiment", alpha=0.5)
    ax2.set_ylabel("Average Daily Sentiment")
    plt.title("Google Stock Price vs News Sentiment")
    fig.tight_layout()
    plt.savefig(os.path.join("static", "price_sentiment.png"))
    plt.close(fig)

    return {
        "logreg_accuracy": float(acc),
        "rf_accuracy": float(rf_acc),
        "rows": int(len(merged)),
        "features": feature_cols,
    }


train_bp = Blueprint("train", __name__)


@train_bp.route("/train", methods=["GET", "POST"])
def train_page():
    """Render the training page with a form to download CSVs and train.
    Accepts two uploaded CSV files (news and prices). Saves them locally,
    runs training, and displays simple metrics on success."""
    message = None
    metrics = None

    TEMPLATE = """
    <html>
      <head>
        <meta charset='utf-8'>
        <title>Update Model</title>
        <style>
          body { font-family: Segoe UI, Arial, sans-serif; margin: 24px; }
          .card { background: #fff; border-radius: 12px; padding: 16px; box-shadow: 0 1px 6px rgba(0,0,0,0.12); max-width: 820px; }
          label { display:block; margin-top:10px; }
          input[type=text] { width:100%; padding:8px; border:1px solid #d1d5db; border-radius:8px; }
          button { margin-top:12px; padding:8px 14px; border:none; border-radius:999px; background:#2563eb; color:#fff; cursor:pointer; }
          .ok { color:#166534; background:#ecfdf3; border:1px solid #bbf7d0; padding:8px; border-radius:8px; }
          .err { color:#b91c1c; background:#fef2f2; border:1px solid #fecaca; padding:8px; border-radius:8px; }
          .metrics { margin-top:12px; }
        </style>
      </head>
      <body>
        <div class="card">
          <h2>Upload CSVs & Train Model</h2>
          <p>Upload <code>Google_Daily_News.csv</code> and <code>googl_daily_prices.csv</code>. After upload, training will run and artifacts update.</p>
          <form method="post" enctype="multipart/form-data">
            <label>News CSV file (Google_Daily_News.csv)</label>
            <input type="file" name="news_file" accept=".csv" required>
            <label>Prices CSV file (googl_daily_prices.csv)</label>
            <input type="file" name="prices_file" accept=".csv" required>
            <button type="submit">Upload & Train</button>
          </form>
          {% if message %}
            <div class="{{ 'ok' if metrics else 'err' }}" style="margin-top:12px;">{{ message }}</div>
          {% endif %}
          {% if metrics %}
            <div class="metrics">
              <p>LogReg accuracy: {{ '%.3f'|format(metrics.logreg_accuracy) }}</p>
              <p>RandomForest accuracy: {{ '%.3f'|format(metrics.rf_accuracy) }}</p>
              <p>Rows used: {{ metrics.rows }}</p>
              <p>Artifacts updated in <code>artifacts/</code>. Chart at <code>/static/price_sentiment.png</code>.</p>
            </div>
          {% endif %}
        </div>
      </body>
    </html>
    """

    if request.method == "POST":
        news_file = request.files.get("news_file")
        prices_file = request.files.get("prices_file")
        try:
            if not news_file or not prices_file:
                raise RuntimeError("Both CSV files are required.")
            news_path = os.path.join("Stock_Sentiment_Project", "Google_Daily_News.csv")
            prices_path = os.path.join("Stock_Sentiment_Project", "googl_daily_prices.csv")
            os.makedirs(os.path.dirname(news_path), exist_ok=True)
            news_file.save(news_path)
            prices_file.save(prices_path)
            m = run_training(prices_path, news_path)
            message = "Training completed and artifacts updated."
            class Metrics:
                def __init__(self, d):
                    self.logreg_accuracy = d.get("logreg_accuracy")
                    self.rf_accuracy = d.get("rf_accuracy")
                    self.rows = d.get("rows")
            metrics = Metrics(m)
        except Exception as e:
            message = f"Error: {e}"
            metrics = None

    return render_template_string(TEMPLATE, message=message, metrics=metrics)
