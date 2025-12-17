
"""
Project Proposal: Stock price movement prediction based on financial news sentiment

Project Overview:
This project aim to predict stock price movements using financial news sentiment analysis.
The purpose of this project is to analyze whether news sentiment has a measurable impact on Google’s stock price movements. 
Using historical stock price data and financial news articles. We will compute sentiment scores and examine how they correlate with real market behavior.
We will develop a simple machine learning model that predicts stock price direction (up or down) based on sentiment indicators, 
and deploy the results in an interactive web application so users can interactively view sentiment and stock price predictions.

Project Objectives:
1. **Data Collection and Preprocessing**:
Dataset: Google Daily Stock Prices (2004–Today) https://www.kaggle.com/datasets/emrekaany/google-daily-stock-prices-2004-today
         Google Financial News Dataset (optional / based on availability) https://www.kaggle.com/datasets/emrekaany/google-googl-financial-news-from-2000-to-today/
Extra data: NewsAPI, public stock price API

2. **Model Development**:
Develop sentiment analysis model and price predictive model

3. **Model Evaluation and Optimization**:
Evaluate model performance using appropriate metrics and optimize the model

4. **Visualization and Reporting**:
Build appropriate visulization charts and write a summary report

5. **Interactive Web App       
Building a streamlit app displaying historical stock price and stock price prediction
List recent top news about google


Scope of Work:
**Phase 1: Requirements Gathering**
- Dateset identification
- Business goal identification
- FE design

**Phase 2: Data Collection and Preprocessing**
•	Load stock price dataset (Open, Close, High, Low, Volume).
•	Load financial news dataset (news title, date).
•	Clean text data (remove punctuation, stopwords, special characters).
•	Align news dates with corresponding stock trading days.

**Phase 3: Model Development**
- Sentiment Analysis model
•	Use a Python NLP library (VADER, TextBlob, or HuggingFace model).
•	Generate sentiment scores for each news headline/article.
•	Categorize sentiment as positive, neutral, or negative.
- Analyze correlation between news sentiment score and stock price movement
- Predictive Modeling (Simple ML)
Convert sentiment into predictive features and build a model to predict price movement (Up/Down) using:
•	Sentiment score
•	Lagged stock returns
•	Volume
•	Other simple features
Possible models:
•	Logistic Regression
•	Random Forest Classifier
•	Support Vector Machine
Goal: Predict whether the next day’s price will go up or down based on sentiment.

**Phase 4: Model Evaluation and Optimization**
- Using appropriate performance metrics (accuracy, precision, F1 score)

**Phase 5: Visualization and Reporting**
- Build appropriate visulization charts using Matplotlib / Seaborn
- Write a summary report

**Phase 6: Deployment: Interactive Web App**
Develop an interactive web interface to present the project results in a simple dashboard and allow users to explore insights through a browser.
The web app will display:
•   Correlation chart: Sentiment scores vs stock price 
•   Stock price + predictions 
•   List top news and sentiment


Deliverables:
•	A Python script includes data cleaning and preprocessing, model development, model evaluation and visualization.
•   An interactive web application.
•	A short report summarizing the findings, challenges, and predictive results.
Expected Outcome
•	Whether strong positive sentiment leads to upward price movement.
•	Whether negative sentiment predicts a drop.
•	How accurately a simple model can sentiment score based on sentiment.
The project does not aim to build a real trading system—just to explore the relationship between news sentiment and stock performance using Python.

Tools & Technologies:
•	Programming language: Python
•	Libraries: Pandas, NumPy, NLTK / VADER / TextBlob (for sentiment), Matplotlib / Seaborn, Scikit-learn
•	Flask (for Interactive web interface)
•	Version Control: Github

Timeline:
Week_1: Phase 1 Requirements Gathering & Phase 2 Data Collection and Preprocessing
Week_2: Phase 3 Model Development
Week_3: Phase 4 Model Evaluation and Optimization & Phase 5 Visualization and Reporting
Week_4: Phase 6 Interactive Web App

Risk:
- Stock prices are influenced by many other factors, and news sentiment alone shows a weak or inconsistent relationship with stock price movements. 
 For example, markets react before news is published, news are not meaningful or fake news, news sentiment doesn't mean investor sentiment.
     
"""