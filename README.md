# Google Stock & News Sentiment Dashboard

An interactive web application that analyzes Google (GOOGL) stock prices alongside news sentiment. This tool uses generic VADER sentiment analysis combined with a Random Forest classifier to predict price movements based on news headlines.

## Features

- **Price vs Sentiment History**: Interactive graph correlating historical stock prices with average daily news sentiment.
- **Sentiment Prediction**: Analyze custom headlines or fetch articles via URL to predict "UP", "DOWN", or "UNCERTAIN" price movements.
- **Recent News Feed**: Automatically fetches recent Google-related news using DuckDuckGo and allows for bulk sentiment analysis.
- **Data Visualization**: Displays the last 10 trading days' data in a tabular format.
- **Model Training**: Interface to retrain the underlying machine learning model with new data.

## Prerequisites

- Python 3.13 or higher
- [uv](https://github.com/astral-sh/uv) (Recommended for dependency management)

## Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd dbu.fall.python.final.groupB
    ```

2.  **Install dependencies**:
    This project uses `uv` for dependency management.
    ```bash
    uv sync
    ```
    To enter the virtual environment:
    ```bash
    source .venv/bin/activate  # On Unix/macOS
    .venv\Scripts\activate     # On Windows
    ```

    *Alternatively, if you prefer pip:*
    ```bash
    pip install .
    ```

## Usage

1.  **Run the Application**:
    ```bash
    python Stock_Sentiment_Project/app.py
    ```

2.  **Access the Dashboard**:
    Open your web browser and navigate to:
    ```
    http://127.0.0.1:5000
    ```

3.  **Explore**:
    - View historical trends in the "Price vs Sentiment" chart.
    - Paste a news headline or URL in section 3 to get a real-time prediction.
    - Select recent news items in section 4 to bulk analyze their sentiment.

## Project Structure

- `Stock_Sentiment_Project/`: Main application source code.
    - `app.py`: Flask application entry point.
    - `templates/`: HTML templates for the web interface.
    - `stock_sentiment_train.py`: Logic for training the ML model.
    - `artifacts/`: Stores trained models (`model.pkl`, `rf_sentiment_classifier.pkl`) and data.
    - `models/`: Additional directory for model storage.
- `daily_sentiment_prices.csv`: Historical data used for graph generation.
- `pyproject.toml`: Project configuration and dependencies.

## Disclaimer

This project is for academic and educational purposes only. The sentiment analysis and price predictions are based on simple models and **should not be used as financial advice** or for real trading decisions.