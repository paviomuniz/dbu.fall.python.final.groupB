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
    stock_path = "google_stocks.csv"
    news_path = "google_news.csv"

    print("Loading CSV files...")
    stocks = pd.read_csv(stock_path)
    news = pd.read_csv(news_path)

    # ------------------------------------------------------------
    # 2) Normalize column names (lowercase -> avoid surprises)
    # ------------------------------------------------------------
    stocks.columns = [c.lower() for c in stocks.columns]
    news.columns = [c.lower() for c in news.columns]

    # Try to detect the right column names
    # Adjust here if your CSVs use different names
    date_col_stock = "date"
    close_col = "close" if "close" in stocks.columns else "adj_close"
    volume_col = "volume"

    date_col_news = "date"
    title_col = "title"

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
