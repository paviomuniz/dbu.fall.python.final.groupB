import os
import pickle
import pandas as pd
import numpy as np
from flask import Blueprint, request, render_template_string
from typing import Dict
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


###Load data
# Load the data from uploaded CSV file into a pandas DataFramw with selected columns
# Load news data

df_news = pd.read_csv(
    os.path.join(BASE_DIR, "Google_News_Data_official.csv"),
    usecols=["uuid", "title", "description", "keywords", "snippet", "published_at"]
)
print("News data loaded successfully.")
print("First 5 rows of the DataFrame:")
df_news.head()

# Load stock price data
df_prices = df_prices = pd.read_csv(
    os.path.join(BASE_DIR, "googl_daily_prices.csv")
)
print("Stock price data loaded successfully.")
print("First 5 rows of df_prices:")
print(df_prices.head())

### Data Preprocessing
# Check data
print("DataFrame Information for df_news:")
df_news.info()

print("\nDataFrame Description:")
df_news.describe()

print("\nDataFrame Information for df_prices:")
df_prices.info()

print("\nDataFrame Description:")
df_prices.describe()


## Clean data
# Remove null data and duplicate data
print("\nNews Data: Number of missing values per column:")
print(df_news.isnull().sum())
df_news = df_news.dropna(subset=["snippet"])
df_news.drop_duplicates(subset=['title','uuid'], keep="first")
print("Null data and uplicates have been removed.")


## News data distribution
df_news['published_at'] = pd.to_datetime(
    df_news['published_at'],
    errors='coerce'   # invalid dates become NaT instead of crashing
)

yearly_counts = (
    df_news
    .groupby(df_news['published_at'].dt.year)
    .size()
    .reset_index(name='total_count')
)
yearly_counts


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
df_news['sentiment_scores'] = df_news['snippet'].apply(analyzer.polarity_scores)
df_news['compound_score'] = df_news['sentiment_scores'].apply(lambda x: x['compound'])

print("Sentiment analysis performed and compound scores added to the DataFrame.")
print("First 5 rows with new sentiment scores:")
print(df_news[['snippet', 'compound_score']].head())

def get_sentiment(score):
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

df_news['sentiment'] = df_news['compound_score'].apply(get_sentiment)
df_news[['snippet', 'compound_score', 'sentiment']].head()

plt.figure(figsize=(8, 6))
sns.countplot(x='sentiment', data=df_news, palette={'Positive': 'green', 'Negative': 'red', 'Neutral': 'blue'}, hue='sentiment', legend=False)
plt.title('Sentiment Distribution of Text Data')
plt.xlabel('Sentiment Category')
plt.ylabel('Number of Snippets')
plt.show()

### Merge News data and stock price data by date
df_news['published_date'] = pd.to_datetime(df_news['published_at'],
                                           errors="coerce",
                                           utc=True ).dt.date
df_prices['price_date'] = pd.to_datetime(df_prices['date']).dt.date

print("\nFirst 5 rows of df_prices with new 'price_date' column:")
df_prices[['date', 'price_date']].head()

merged_df = pd.merge(df_news, df_prices, left_on='published_date', right_on='price_date', how='left')

print("First 5 rows of merged_df:")
merged_df.head()

daily_sentiment_prices = merged_df.groupby('published_date').agg(
    avg_sentiment=('compound_score', 'mean'),
    avg_close=('close', 'mean')
).reset_index()

print("Aggregated daily sentiment and close prices:")
daily_sentiment_prices.head()
daily_sentiment_prices['published_date'] = pd.to_datetime(daily_sentiment_prices['published_date'])
daily_sentiment_prices.to_csv('daily_sentiment_prices.csv', index = False)
plt.figure(figsize=(50, 7))

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

## Visulize relationship between news sentiment score and stock price
# Graph 1
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
plt.show()

#Graph 2
plt.figure(figsize=(50, 7))
daily_sentiment_prices['moving_sentiment'] = (
    daily_sentiment_prices.sort_values('published_date')['avg_sentiment'].rolling(window=7, min_periods=1).mean()
)

sns.lineplot(x='published_date', y='moving_sentiment', data=daily_sentiment_prices, label='Moving Sentiment', color='blue')

ax2 = plt.twinx()
sns.lineplot(x='published_date', y='avg_close', data=daily_sentiment_prices, label='Average Close Price', color='red', ax=ax2)
plt.title('Stock Price vs 7D Moving Average Sentiment Score')
plt.xlabel('Date')
plt.ylabel('Moving Average Sentiment Score', color='blue')
ax2.set_ylabel('Stock Close Price', color='red')
plt.grid(True)
plt.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.show()


#Graph 3
plt.figure(figsize=(50, 7))

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
plt.show()


###Linear Regression Model and Random Forest Regression Model
##Prepare data for linear modeling
print("___________________________")
print("Linear Regression & Random Forest Regression")
daily_sentiment_prices = daily_sentiment_prices.sort_values(by='published_date', ascending=True)

# Create a new column 'previous_close' by shifting 'avg_close' by one day.
daily_sentiment_prices['previous_close'] = daily_sentiment_prices['avg_close'].shift(1)
daily_sentiment_prices.dropna(inplace=True)
X = daily_sentiment_prices[['avg_sentiment', 'previous_close']]
y = daily_sentiment_prices['avg_close']

print("First 5 rows of the prepared DataFrame with 'previous_close' column:")
daily_sentiment_prices.head()
print("\nShape of X:", X.shape)
print("Shape of y:", y.shape)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Data split into training and testing sets.")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

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

### Visalize Model Predictions
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
plt.show()


### Random Forest Classifier model
print("___________________________")
print("Random Forest Classification Model")
# Create a new target variable 'price_change_sign'
daily_sentiment_prices['price_change_sign'] = np.where(
    daily_sentiment_prices['daily_return'] > 0, 1, 0
)
# Define the independent variable X
X = daily_sentiment_prices[['avg_sentiment']]
# Define the dependent variable y
y = daily_sentiment_prices['price_change_sign']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Data split into training and testing sets.")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# Initialize and train the Random Forest Classifier model
random_forest_classifier = RandomForestClassifier(random_state=42)
random_forest_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred_rf_classifier = random_forest_classifier.predict(X_test)

# Evaluate the model
accuracy_rf = accuracy_score(y_test, y_pred_rf_classifier)
precision_rf = precision_score(y_test, y_pred_rf_classifier)
recall_rf = recall_score(y_test, y_pred_rf_classifier)
f1_rf = f1_score(y_test, y_pred_rf_classifier)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf_classifier)

print("Random Forest Classifier Model Training and Evaluation:")
print(f"Accuracy: {accuracy_rf:.2f}")
print(f"Precision: {precision_rf:.2f}")
print(f"Recall: {recall_rf:.2f}")
print(f"F1-score: {f1_rf:.2f}")
print("Confusion Matrix:")
print(conf_matrix_rf)

# Initialize and train the Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Make predictions on the test set (continuous values)
y_pred_linear_continuous = linear_model.predict(X_test)

# Convert continuous predictions to binary outcomes using a threshold of 0.5
y_pred_linear_binary = np.where(y_pred_linear_continuous >= 0.5, 1, 0)

# Evaluate the model using classification metrics
accuracy_linear = accuracy_score(y_test, y_pred_linear_binary)
precision_linear = precision_score(y_test, y_pred_linear_binary)
recall_linear = recall_score(y_test, y_pred_linear_binary)
f1_linear = f1_score(y_test, y_pred_linear_binary)

print("Linear Regression Model Training and Evaluation for Binary Classification:")
print(f"Accuracy: {accuracy_linear:.2f}")
print(f"Precision: {precision_linear:.2f}")
print(f"Recall: {recall_linear:.2f}")
print(f"F1-score: {f1_linear:.2f}")


plt.figure(figsize=(12, 5))

# Confusion Matrix for Linear Regression Model
plt.subplot(1, 2, 1)
cm_linear = confusion_matrix(y_test, y_pred_linear_binary)
sns.heatmap(cm_linear, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Linear Regression Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Confusion Matrix for Random Forest Classifier Model
plt.subplot(1, 2, 2)
sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Random Forest Classifier Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.tight_layout()
plt.show()

import pickle

MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

with open(os.path.join(MODEL_DIR, "rf_sentiment_classifier.pkl"), "wb") as f:
    pickle.dump(random_forest_classifier, f)

print("Random Forest classifier saved successfully.")

#     from sklearn.model_selection import train_test_split
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42, shuffle=True
#     )

#     from sklearn.preprocessing import StandardScaler
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)

#     from sklearn.linear_model import LogisticRegression
#     model = LogisticRegression(max_iter=1000)
#     model.fit(X_train_scaled, y_train)

#     from sklearn.ensemble import RandomForestClassifier
#     rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
#     rf_model.fit(X_train, y_train)

#     from sklearn.metrics import accuracy_score
#     y_pred = model.predict(X_test_scaled)
#     acc = accuracy_score(y_test, y_pred)

#     rf_pred = rf_model.predict(X_test)
#     rf_acc = accuracy_score(y_test, rf_pred)

#     os.makedirs("artifacts", exist_ok=True)
#     with open(os.path.join("artifacts", "model.pkl"), "wb") as f:
#         pickle.dump(model, f)
#     with open(os.path.join("artifacts", "scaler.pkl"), "wb") as f:
#         pickle.dump(scaler, f)
#     merged.to_csv(os.path.join("artifacts", "data_with_sentiment.csv"), index=False)

#     import matplotlib.pyplot as plt
#     os.makedirs("static", exist_ok=True)
#     fig, ax1 = plt.subplots(figsize=(10, 5))
#     ax1.plot(merged[date_col_stock], merged[close_col], label="Close Price")
#     ax1.set_xlabel("Date")
#     ax1.set_ylabel("Close Price")

#     ax2 = ax1.twinx()
#     ax2.plot(merged[date_col_stock], merged["sentiment_mean"], label="Sentiment", alpha=0.5)
#     ax2.set_ylabel("Average Daily Sentiment")
#     plt.title("Google Stock Price vs News Sentiment")
#     fig.tight_layout()
#     plt.savefig(os.path.join("static", "price_sentiment.png"))
#     plt.close(fig)

#     return {
#         "logreg_accuracy": float(acc),
#         "rf_accuracy": float(rf_acc),
#         "rows": int(len(merged)),
#         "features": feature_cols,
#     }


# train_bp = Blueprint("train", __name__)


# @train_bp.route("/train", methods=["GET", "POST"])
# def train_page():
#     """Render the training page with a form to download CSVs and train.
#     Accepts two uploaded CSV files (news and prices). Saves them locally,
#     runs training, and displays simple metrics on success."""
#     message = None
#     metrics = None

#     TEMPLATE = """
#     <html>
#       <head>
#         <meta charset='utf-8'>
#         <title>Update Model</title>
#         <style>
#           body { font-family: Segoe UI, Arial, sans-serif; margin: 24px; }
#           .card { background: #fff; border-radius: 12px; padding: 16px; box-shadow: 0 1px 6px rgba(0,0,0,0.12); max-width: 820px; }
#           label { display:block; margin-top:10px; }
#           input[type=text] { width:100%; padding:8px; border:1px solid #d1d5db; border-radius:8px; }
#           button { margin-top:12px; padding:8px 14px; border:none; border-radius:999px; background:#2563eb; color:#fff; cursor:pointer; }
#           .ok { color:#166534; background:#ecfdf3; border:1px solid #bbf7d0; padding:8px; border-radius:8px; }
#           .err { color:#b91c1c; background:#fef2f2; border:1px solid #fecaca; padding:8px; border-radius:8px; }
#           .metrics { margin-top:12px; }
#         </style>
#       </head>
#       <body>
#         <div class="card">
#           <h2>Upload CSVs & Train Model</h2>
#           <p>Upload <code>Google_Daily_News.csv</code> and <code>googl_daily_prices.csv</code>. After upload, training will run and artifacts update.</p>
#           <form method="post" enctype="multipart/form-data">
#             <label>News CSV file (Google_Daily_News.csv)</label>
#             <input type="file" name="news_file" accept=".csv" required>
#             <label>Prices CSV file (googl_daily_prices.csv)</label>
#             <input type="file" name="prices_file" accept=".csv" required>
#             <button type="submit">Upload & Train</button>
#           </form>
#           {% if message %}
#             <div class="{{ 'ok' if metrics else 'err' }}" style="margin-top:12px;">{{ message }}</div>
#           {% endif %}
#           {% if metrics %}
#             <div class="metrics">
#               <p>LogReg accuracy: {{ '%.3f'|format(metrics.logreg_accuracy) }}</p>
#               <p>RandomForest accuracy: {{ '%.3f'|format(metrics.rf_accuracy) }}</p>
#               <p>Rows used: {{ metrics.rows }}</p>
#               <p>Artifacts updated in <code>artifacts/</code>. Chart at <code>/static/price_sentiment.png</code>.</p>
#             </div>
#           {% endif %}
#         </div>
#       </body>
#     </html>
#     """

#     if request.method == "POST":
#         news_file = request.files.get("news_file")
#         prices_file = request.files.get("prices_file")
#         try:
#             if not news_file or not prices_file:
#                 raise RuntimeError("Both CSV files are required.")
#             news_path = os.path.join("Stock_Sentiment_Project", "Google_Daily_News.csv")
#             prices_path = os.path.join("Stock_Sentiment_Project", "googl_daily_prices.csv")
#             os.makedirs(os.path.dirname(news_path), exist_ok=True)
#             news_file.save(news_path)
#             prices_file.save(prices_path)
#             m = run_training(prices_path, news_path)
#             message = "Training completed and artifacts updated."
#             class Metrics:
#                 def __init__(self, d):
#                     self.logreg_accuracy = d.get("logreg_accuracy")
#                     self.rf_accuracy = d.get("rf_accuracy")
#                     self.rows = d.get("rows")
#             metrics = Metrics(m)
#         except Exception as e:
#             message = f"Error: {e}"
#             metrics = None

#     return render_template_string(TEMPLATE, message=message, metrics=metrics)
