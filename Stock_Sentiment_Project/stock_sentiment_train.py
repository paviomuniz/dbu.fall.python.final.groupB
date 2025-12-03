import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer


def main():
    # ------------------------------------------------------------
    # 1) Load datasets
    # ------------------------------------------------------------
    # TODO: change these file names to match your actual CSV files
    stock_path = "googl_daily_prices.csv"
    news_path = "google_Daily_news.csv"

    print("Loading CSV files...")
    stocks = pd.read_csv(stock_path)
    news = pd.read_csv(news_path)

    # ------------------------------------------------------------
    # 2) Normalize column names (lowercase -> avoid surprises)
    # ------------------------------------------------------------
    stocks.columns = [c.lower() for c in stocks.columns]
    news.columns = [c.lower() for c in news.columns]

    print("Stock columns:", list(stocks.columns))
    print("News columns:", list(news.columns))

    # --- stock date column ---
    stock_date_candidates = [c for c in stocks.columns if "date" in c]
    if not stock_date_candidates:
        raise ValueError("Could not find a date column in the stock CSV.")
    date_col_stock = stock_date_candidates[0]

    # --- stock close price column ---
    close_candidates = [c for c in stocks.columns if "close" in c]
    if not close_candidates:
        raise ValueError("Could not find a close/adj close column in the stock CSV.")
    close_col = close_candidates[0]

    # --- stock volume column ---
    volume_candidates = [c for c in stocks.columns if "volume" in c]
    if not volume_candidates:
        raise ValueError("Could not find a volume column in the stock CSV.")
    volume_col = volume_candidates[0]

    # --- news date column ---
    news_date_candidates = [
        c for c in news.columns
        if "date" in c or "time" in c or "publish" in c
    ]
    if not news_date_candidates:
        raise ValueError("Could not find a date/published column in the news CSV.")
    date_col_news = news_date_candidates[0]

    # --- news title/headline column ---
    news_title_candidates = [
        c for c in news.columns
        if "title" in c or "headline" in c or "news" in c
    ]
    if not news_title_candidates:
        raise ValueError("Could not find a title/headline column in the news CSV.")
    title_col = news_title_candidates[0]

    print(f"Using stock date column: {date_col_stock}")
    print(f"Using stock close column: {close_col}")
    print(f"Using stock volume column: {volume_col}")
    print(f"Using news date column: {date_col_news}")
    print(f"Using news title column: {title_col}")

    # ------------------------------------------------------------
    # 3) Parse dates and sort
    # ------------------------------------------------------------
    print("Parsing dates...")
    stocks[date_col_stock] = pd.to_datetime(stocks[date_col_stock])
    stocks = stocks.sort_values(date_col_stock)

    news[date_col_news] = pd.to_datetime(news[date_col_news])

    # ------------------------------------------------------------
    # 4) Create stock return + target (next-day up or down)
    # ------------------------------------------------------------
    print("Creating target variable (next-day up/down)...")
    # Daily return (percentage change)
    stocks["return"] = stocks[close_col].pct_change()

    # Target: 1 if next day's close > today's close, else 0
    stocks["target"] = (stocks[close_col].shift(-1) > stocks[close_col]).astype(int)

    # Drop rows with missing values in these columns
    stocks = stocks.dropna(subset=["return", "target"])

    # ------------------------------------------------------------
    # 5) Prepare news sentiment (VADER)
    # ------------------------------------------------------------
    print("Computing sentiment for news...")
    nltk.download("vader_lexicon", quiet=True)
    sia = SentimentIntensityAnalyzer()

    # Some titles might be NaN -> convert to string
    news["sentiment"] = news[title_col].astype(str).apply(
        lambda text: sia.polarity_scores(text)["compound"]
    )

    # Aggregate sentiment per day (average sentiment for that day)
    news_daily = (
        news.groupby(news[date_col_news].dt.date)["sentiment"]
        .mean()
        .reset_index()
        .rename(columns={date_col_news: "date_only", "sentiment": "sentiment_mean"})
    )

    # ------------------------------------------------------------
    # 6) Align dates between stocks and news
    # ------------------------------------------------------------
    print("Merging stock data with daily sentiment...")
    stocks["date_only"] = stocks[date_col_stock].dt.date

    merged = pd.merge(
        stocks,
        news_daily,
        how="left",
        left_on="date_only",
        right_on="date_only",
    )

    # If some days have no news, set sentiment to 0 (neutral)
    merged["sentiment_mean"] = merged["sentiment_mean"].fillna(0.0)

    # ------------------------------------------------------------
    # 7) Build features (X) and labels (y)
    # ------------------------------------------------------------
    print("Building features...")
    feature_cols = ["sentiment_mean", "return", volume_col]
    X = merged[feature_cols].values
    y = merged["target"].values

    # Remove rows with NaN in features
    mask = ~np.isnan(X).any(axis=1)
    X = X[mask]
    y = y[mask]
    merged = merged.loc[mask].reset_index(drop=True)
  # ------------------------------------------------------------
    # 8) Train / test split
    # ------------------------------------------------------------
    print("Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    # ------------------------------------------------------------
    # 9) Scale numeric features
    # ------------------------------------------------------------
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ------------------------------------------------------------
    # 10) Train Logistic Regression model
    # ------------------------------------------------------------
    print("Training model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # ------------------------------------------------------------
    # 11) Evaluate model
    # ------------------------------------------------------------
    print("Evaluating model...")
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    print("\n=== Model Evaluation ===")
    print(f"Accuracy: {acc:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # ------------------------------------------------------------
    # 12) Save model, scaler, and merged data
    # ------------------------------------------------------------
    print("\nSaving artifacts...")
    os.makedirs("artifacts", exist_ok=True)

    with open(os.path.join("artifacts", "model.pkl"), "wb") as f:
        pickle.dump(model, f)

    with open(os.path.join("artifacts", "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    merged.to_csv(os.path.join("artifacts", "data_with_sentiment.csv"), index=False)

    # ------------------------------------------------------------
    # 13) Create a chart: price vs sentiment
    # ------------------------------------------------------------
    print("Creating price vs sentiment chart...")
    os.makedirs("static", exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(merged[date_col_stock], merged[close_col], label="Close Price")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Close Price")

    ax2 = ax1.twinx()
    ax2.plot(
        merged[date_col_stock],
        merged["sentiment_mean"],
        label="Sentiment",
        alpha=0.5,
    )
    ax2.set_ylabel("Average Daily Sentiment")

    plt.title("Google Stock Price vs News Sentiment")
    fig.tight_layout()
    plt.savefig(os.path.join("static", "price_sentiment.png"))
    plt.close(fig)

    print("\n✅ Done!")
    print("Artifacts saved in the 'artifacts' folder.")
    print("Chart saved as 'static/price_sentiment.png'.")


if __name__ == "__main__":
    main()