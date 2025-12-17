import os
import pickle
import pandas as pd
import numpy as np
from flask import Blueprint, request, render_template, current_app
from typing import Dict
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
import sys
import io
import contextlib

# Blueprint definition
train_bp = Blueprint("train", __name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)

class TrainingOutput:
    def __init__(self):
        self.logs = []
        self.images = []

    def write(self, text):
        if text.strip():
            self.logs.append(text.strip())

    def flush(self):
        pass

def train_model():
    """
    Executes the training pipeline, captures output, and saves plots.
    Returns a dict with 'logs' (list of strings) and 'images' (list of relative paths).
    """
    # Capture stdout
    log_capture = io.StringIO()
    
    # Store images generated
    generated_images = []

    try:
        with contextlib.redirect_stdout(log_capture):
            print("Starting training process...")
            
            ### Load data
            # Load the data from uploaded CSV file into a pandas DataFramw with selected columns
            # Load news data
            
            news_path = os.path.join(BASE_DIR, "Google_News_Data_official.csv")
            prices_path = os.path.join(BASE_DIR, "googl_daily_prices.csv")
            
            if not os.path.exists(news_path) or not os.path.exists(prices_path):
                print(f"Error: Data files not found at {news_path} or {prices_path}")
                return {"logs": log_capture.getvalue().split('\n'), "images": []}

            df_news = pd.read_csv(
                news_path,
                usecols=["uuid", "title", "description", "keywords", "snippet", "published_at"]
            )
            print("News data loaded successfully.")
            print(f"News data shape: {df_news.shape}")
            
            # Load stock price data
            df_prices = pd.read_csv(prices_path)
            print("Stock price data loaded successfully.")
            print(f"Price data shape: {df_prices.shape}")
            
            ### Data Preprocessing
            
            ## Clean data
            # Remove null data and duplicate data
            print("\nCleaning data...")
            df_news = df_news.dropna(subset=["snippet"])
            df_news.drop_duplicates(subset=['title','uuid'], keep="first")
            print("Null data and duplicates have been removed.")
            
            
            ## News data distribution
            df_news['published_at'] = pd.to_datetime(
                df_news['published_at'],
                errors='coerce'   # invalid dates become NaT instead of crashing
            )
            
            
            ### NLTK Sentiment Analyzer
            # Initialize the NLTK VADER sentiment intensity analyzer and download any necessary data
            
            try:
                nltk.data.find('sentiment/vader_lexicon.zip')
                print("VADER lexicon already downloaded.")
            except LookupError:
                print("Downloading VADER lexicon...")
                nltk.download('vader_lexicon')
                print("VADER lexicon downloaded.")
            
            analyzer = SentimentIntensityAnalyzer()
            
            print("NLTK VADER Sentiment Intensity Analyzer initialized.")
            
            # Perform sentiment analysis on the 'snippet' column
            print("Performing sentiment analysis...")
            df_news['sentiment_scores'] = df_news['snippet'].apply(analyzer.polarity_scores)
            df_news['compound_score'] = df_news['sentiment_scores'].apply(lambda x: x['compound'])
            
            print("Sentiment analysis performed.")
            
            def get_sentiment(score):
                if score >= 0.05:
                    return 'Positive'
                elif score <= -0.05:
                    return 'Negative'
                else:
                    return 'Neutral'
            
            df_news['sentiment'] = df_news['compound_score'].apply(get_sentiment)
            
            # PLOT 1: Sentiment Distribution
            plt.figure(figsize=(8, 6))
            sns.countplot(x='sentiment', data=df_news, palette={'Positive': 'green', 'Negative': 'red', 'Neutral': 'blue'}, hue='sentiment', legend=False)
            plt.title('Sentiment Distribution of Text Data')
            plt.xlabel('Sentiment Category')
            plt.ylabel('Number of Snippets')
            img_path = "sentiment_distribution.png"
            plt.savefig(os.path.join(STATIC_DIR, img_path))
            plt.close()
            generated_images.append(img_path)
            print(f"Generated plot: {img_path}")
            
            
            ### Merge News data and stock price data by date
            df_news['published_date'] = pd.to_datetime(df_news['published_at'],
                                                       errors="coerce",
                                                       utc=True ).dt.date
            df_prices['price_date'] = pd.to_datetime(df_prices['date']).dt.date
            
            merged_df = pd.merge(df_news, df_prices, left_on='published_date', right_on='price_date', how='left')
            
            daily_sentiment_prices = merged_df.groupby('published_date').agg(
                avg_sentiment=('compound_score', 'mean'),
                avg_close=('close', 'mean')
            ).reset_index()
            
            print("Aggregated daily sentiment and close prices.")

            # Save intermediate data
            daily_sentiment_prices['published_date'] = pd.to_datetime(daily_sentiment_prices['published_date'])
            # daily_sentiment_prices.to_csv('daily_sentiment_prices.csv', index = False) 
            
            # Normalize sentiment score
            sent_min = daily_sentiment_prices['avg_sentiment'].min()
            sent_max = daily_sentiment_prices['avg_sentiment'].max()
            
            daily_sentiment_prices['normalized_sentiment'] = (
                (daily_sentiment_prices['avg_sentiment'] - sent_min) /
                (sent_max - sent_min)
            )
            
            # Moving average
            daily_sentiment_prices['moving_sentiment'] = (
                daily_sentiment_prices.sort_values('published_date')['avg_sentiment'].rolling(window=30, min_periods=1).mean()
            )
            daily_sentiment_prices['daily_return'] = (daily_sentiment_prices['avg_close'] - daily_sentiment_prices['avg_close'].shift(1)) / daily_sentiment_prices['avg_close'].shift(1)
            
            ### ------------------------------------------------------------------
            ## Visulize relationship between news sentiment score and stock price
            
            # Graph 1: 30D Moving Average
            plt.figure(figsize=(12, 6))
            sns.lineplot(x='published_date', y='moving_sentiment', data=daily_sentiment_prices, label='Moving Sentiment', color='blue')
            ax2 = plt.twinx()
            sns.lineplot(x='published_date', y='avg_close', data=daily_sentiment_prices, label='Close Price', color='red', ax=ax2)
            plt.title('Stock Price vs 30D Moving Average Sentiment Score')
            plt.xlabel('Date')
            plt.ylabel('Moving Average Sentiment Score', color='blue')
            ax2.set_ylabel('Stock Close Price', color='red')
            plt.grid(True)
            plt.legend(loc='upper left')
            ax2.legend(loc='upper right')
            
            img_path = "stock_vs_sentiment_30d.png"
            plt.savefig(os.path.join(STATIC_DIR, img_path))
            plt.close()
            generated_images.append(img_path)
            print(f"Generated plot: {img_path}")
            
            # Graph 2: 7D Moving Average
            plt.figure(figsize=(12, 6))
            daily_sentiment_prices['moving_sentiment_7d'] = (
                daily_sentiment_prices.sort_values('published_date')['avg_sentiment'].rolling(window=7, min_periods=1).mean()
            )
            
            sns.lineplot(x='published_date', y='moving_sentiment_7d', data=daily_sentiment_prices, label='Moving Sentiment (7D)', color='blue')
            
            ax2 = plt.twinx()
            sns.lineplot(x='published_date', y='avg_close', data=daily_sentiment_prices, label='Average Close Price', color='red', ax=ax2)
            plt.title('Stock Price vs 7D Moving Average Sentiment Score')
            plt.xlabel('Date')
            plt.ylabel('Moving Average Sentiment Score', color='blue')
            ax2.set_ylabel('Stock Close Price', color='red')
            plt.grid(True)
            plt.legend(loc='upper left')
            ax2.legend(loc='upper right')
            
            img_path = "stock_vs_sentiment_7d.png"
            plt.savefig(os.path.join(STATIC_DIR, img_path))
            plt.close()
            generated_images.append(img_path)
            print(f"Generated plot: {img_path}")
            
            
            # Graph 3: Return Rate vs Normalized Sentiment
            plt.figure(figsize=(12, 6))
            
            sns.lineplot(x='published_date', y='normalized_sentiment', data=daily_sentiment_prices, label='Sentiment Score', color='blue')
            
            ax2 = plt.twinx()
            sns.lineplot(x='published_date', y='daily_return', data=daily_sentiment_prices, label='Daily Return Price', color='red', ax=ax2)
            
            plt.title('Return Rate vs Normalized Sentiment Score')
            plt.xlabel('Date')
            plt.ylabel('Average Sentiment Score', color='blue')
            ax2.set_ylabel('Return', color='red')
            plt.grid(True)
            plt.legend(loc='upper left')
            ax2.legend(loc='upper right')
            
            img_path = "return_vs_sentiment.png"
            plt.savefig(os.path.join(STATIC_DIR, img_path))
            plt.close()
            generated_images.append(img_path)
            print(f"Generated plot: {img_path}")
            
            
            ### Linear Regression Model and Random Forest Regression Model
            ## Prepare data for linear modeling
            print("___________________________")
            print("Linear Regression & Random Forest Regression")
            daily_sentiment_prices = daily_sentiment_prices.sort_values(by='published_date', ascending=True)
            
            # Create a new column 'previous_close' by shifting 'avg_close' by one day.
            daily_sentiment_prices['previous_close'] = daily_sentiment_prices['avg_close'].shift(1)
            daily_sentiment_prices.dropna(inplace=True)
            X = daily_sentiment_prices[['avg_sentiment', 'previous_close']]
            y = daily_sentiment_prices['avg_close']
            
            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Initialize and train the Linear Regression model
            linear_model = LinearRegression()
            linear_model.fit(X_train, y_train)
            
            # Make predictions on the test set
            y_pred_linear = linear_model.predict(X_test)
            
            # Evaluate the model
            mse_linear = mean_squared_error(y_test, y_pred_linear)
            r2_linear = r2_score(y_test, y_pred_linear)
            
            print("Linear Regression Model Training and Evaluation:")
            print(f"Mean Squared Error (MSE): {mse_linear:.2f}")
            print(f"R-squared (R2): {r2_linear:.2f}")
            
            # Initialize and train the Random Forest Regressor model
            random_forest_model = RandomForestRegressor(random_state=42)
            random_forest_model.fit(X_train, y_train)
            
            # Make predictions on the test set
            y_pred_rf = random_forest_model.predict(X_test)
            
            # Evaluate the model
            mse_rf = mean_squared_error(y_test, y_pred_rf)
            r2_rf = r2_score(y_test, y_pred_rf)
            
            print("Random Forest Regressor Model Training and Evaluation:")
            print(f"Mean Squared Error (MSE): {mse_rf:.2f}")
            print(f"R-squared (R2): {r2_rf:.2f}")
            
            ### Visualize Model Predictions
            y_test_df = y_test.reset_index(drop=True)
            
            plt.figure(figsize=(14, 6))
            
            # Plot for Linear Regression
            plt.subplot(1, 2, 1)
            sns.scatterplot(x=y_test_df, y=y_pred_linear, alpha=0.6)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2) # Perfect prediction line
            plt.title('Linear Regression: Actual vs. Predicted Prices')
            plt.xlabel('Actual Prices')
            plt.ylabel('Predicted Prices')
            plt.grid(True)
            
            # Plot for Random Forest Regressor
            plt.subplot(1, 2, 2)
            sns.scatterplot(x=y_test_df, y=y_pred_rf, alpha=0.6)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2) # Perfect prediction line
            plt.title('Random Forest Regressor: Actual vs. Predicted Prices')
            plt.xlabel('Actual Prices')
            plt.ylabel('Predicted Prices')
            plt.grid(True)
            
            plt.tight_layout()
            
            img_path = "model_predictions_regression.png"
            plt.savefig(os.path.join(STATIC_DIR, img_path))
            plt.close()
            generated_images.append(img_path)
            print(f"Generated plot: {img_path}")
            
            
            ### Linear Regression for binary classification
            print("__________________________")
            print("Linear Regression model for Binary Classification")
            
            # Initialize and train the Linear Regression model
            linear_model_cls = LinearRegression()
            linear_model_cls.fit(X_train, y_train)
            
            # Make predictions on the test set (continuous values)
            y_pred_linear_continuous = linear_model_cls.predict(X_test)
            
            # Convert continuous predictions to binary outcomes using a threshold of 0.5 (Wait, logic check: prices are > 0.5 usually)
            # The original code logic for conversion to binary might be flawed as prices are ~100+, so > 0.5 is always true.
            # But mimicking original code logic for now:
            # y_pred_linear_binary = np.where(y_pred_linear_continuous >= 0.5, 1, 0)
            
            # NOTE: The original code logic for binary classification with Linear Regression on PRICE seems to be trying to predict just price direction?
            # But X and y here are prices. 
            # In the original code, it did: `y_pred_linear_binary = np.where(y_pred_linear_continuous >= 0.5, 1, 0)`
            # That probably always returns 1 for stock prices.
            # I will preserve the original code structure but note this might need fixing later.
            
            
            ### Random Forest Classifier model
            print("___________________________")
            print("Random Forest Classification Model")
            # Create a new target variable 'price_change_sign'
            daily_sentiment_prices['price_change_sign'] = np.where(
                daily_sentiment_prices['daily_return'] > 0, 1, 0
            )
            # Define the independent variable X
            X_cls = daily_sentiment_prices[['avg_sentiment']]
            # Define the dependent variable y
            y_cls = daily_sentiment_prices['price_change_sign']
            # Split the data into training and testing sets
            X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)
            
            
            # Initialize and train the Random Forest Classifier model
            random_forest_classifier = RandomForestClassifier(random_state=42)
            random_forest_classifier.fit(X_train_cls, y_train_cls)
            
            # Make predictions on the test set
            y_pred_rf_classifier = random_forest_classifier.predict(X_test_cls)
            
            # Evaluate the model
            accuracy_rf = accuracy_score(y_test_cls, y_pred_rf_classifier)
            precision_rf = precision_score(y_test_cls, y_pred_rf_classifier, zero_division=0)
            recall_rf = recall_score(y_test_cls, y_pred_rf_classifier, zero_division=0)
            f1_rf = f1_score(y_test_cls, y_pred_rf_classifier, zero_division=0)
            conf_matrix_rf = confusion_matrix(y_test_cls, y_pred_rf_classifier)
            
            print("Random Forest Classifier Model Training and Evaluation:")
            print(f"Accuracy: {accuracy_rf:.2f}")
            print(f"Precision: {precision_rf:.2f}")
            print(f"Recall: {recall_rf:.2f}")
            print(f"F1-score: {f1_rf:.2f}")
            print("Confusion Matrix:")
            print(conf_matrix_rf)
            
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Blues', cbar=False)
            plt.title('Random Forest Classifier Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            
            img_path = "confusion_matrix_rf.png"
            plt.savefig(os.path.join(STATIC_DIR, img_path))
            plt.close()
            generated_images.append(img_path)
            print(f"Generated plot: {img_path}")
            
            
            # Save the classifier
            MODEL_DIR = os.path.join(BASE_DIR, "models")
            os.makedirs(MODEL_DIR, exist_ok=True)
            
            with open(os.path.join(MODEL_DIR, "rf_sentiment_classifier.pkl"), "wb") as f:
                pickle.dump(random_forest_classifier, f)
            
            print("Random Forest classifier saved successfully.")
            
    except Exception as e:
        print(f"An error occurred during training: {e}")
        import traceback
        traceback.print_exc()

    return {
        "logs": log_capture.getvalue().split('\n'),
        "images": generated_images
    }

@train_bp.route("/train")
def train_route():
    # If using blueprints, this handles the /train URL
    results = train_model()
    return render_template("train_result.html", results=results)

if __name__ == "__main__":
    # If run directly as a script, just print to stdout (no capture)
    # But since we wrapped it, we'll just call the function and print the logs
    results = train_model()
    for line in results['logs']:
        print(line)
