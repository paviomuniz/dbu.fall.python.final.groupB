
"""
Project Proposal: Analyzing News Sentiment vs. Google Stock Price Movements
1. Project Title
Predicting Stock Price Movements Using News Sentiment Analysis (Google Stock Dataset)

2. The purpose of this project is to analyze whether news sentiment has a measurable impact on Google’s daily stock price movements. Using historical stock price data and financial news articles, we will compute sentiment scores and examine how they correlate with real market behavior.

Additionally, we will experiment with a simple machine learning model that predicts short-term stock price direction (up or down) based on sentiment indicators. Finally, we plan to deploy the results in a Flask web application so users can interactively view sentiment, charts, and model predictions.

3. Datasets to Be Used

1.	Google Daily Stock Prices (2004–Today)
                              
https://www.kaggle.com/datasets/emrekaany/google-daily-stock-prices-2004-today

2.	Google Financial News Dataset (optional / based on availability)
https://www.kaggle.com/datasets/emrekaany/google-googl-financial-news-from-2000-to-today/

4. Scope of Work
This is what we intend to accomplish:
A. Data Collection & Cleaning
•	Load stock price dataset (Open, Close, High, Low, Volume).
•	Load financial news dataset (news title, date).
•	Clean text data (remove punctuation, stopwords, special characters).
•	Align news dates with corresponding stock trading days.

B. Sentiment Analysis
•	Use a Python NLP library (VADER, TextBlob, or HuggingFace model).
•	Generate sentiment scores for each news headline/article.
•	Categorize sentiment as positive, neutral, or negative.

C. Exploratory Data Analysis
We will analyze:
•	Does positive news correlate with price increases?
•	Does negative news correlate with price drops?
•	How strong is the correlation between sentiment score and daily returns?
•	Plot sentiment vs. stock price over time.

D. Predictive Modeling (Simple ML)
We will build a basic model to predict price movement (Up/Down) using:
•	Sentiment score
•	Lagged stock returns
•	Volume
•	Other simple features
Possible models:
•	Logistic Regression
•	Random Forest Classifier
•	Support Vector Machine
Goal: Predict whether the next day’s price will go up or down based on sentiment.

E. Deliverables
Our final project deliverables will include:
•	A Python script/Visual Studio explaining each step.
•	Visualizations (sentiment over time, correlation graphs, prediction accuracy).
•   A Flask web application to display results.
•	A short report summarizing the findings, challenges, and predictive results.

F. Flask Web Application

We will develop a Flask web interface to present the project results in a simple dashboard.
The web app will display:

•   Sentiment scores
•   Stock price charts
•   Correlation visualizations
•   Model predictions (Up/Down)
•   Optionally: Upload or paste custom news headlines for sentiment evaluation

This adds an interactive component and allows users to explore insights through a browser.

5. Expected Outcome
We expect to determine:
•	Whether strong positive sentiment leads to upward price movement.
•	Whether negative sentiment predicts a drop.
•	How accurately a simple model can predict price direction based on sentiment.
The project does not aim to build a real trading system—just to explore the relationship between news sentiment and stock performance using Python.

6. Tools & Technologies
•	Python
•	Pandas, NumPy
•	NLTK / VADER / TextBlob (for sentiment)
•	Matplotlib / Seaborn
•	Scikit-learn
•	Flask (for Interactive web interface)

"""