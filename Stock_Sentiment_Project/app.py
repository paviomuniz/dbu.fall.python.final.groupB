from flask import Flask, render_template, request
from stock_sentiment_train import train_bp
import pandas as pd
import pickle
import os
from datetime import datetime
import urllib.request
import urllib.parse
import ssl
import re
import html

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

try:
    with open(os.path.join(MODEL_DIR, "rf_sentiment_classifier.pkl"), "rb") as f:
        rf_model = pickle.load(f)
except Exception:
    rf_model = None

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

def duckduckgo_recent_news(query: str, days: int = 3, max_results: int = 50):
    try:
        import time
        import json
        import urllib.parse
        now = int(time.time())
        threshold = now - days * 86400
        ctx = ssl.create_default_context()
        ua = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"}
        q = urllib.parse.quote(query)
        html_page = urllib.request.urlopen(urllib.request.Request(f"https://duckduckgo.com/?q={q}&iar=news&ia=news", headers=ua), context=ctx, timeout=10).read().decode("utf-8", "ignore")
        m = re.search(r"vqd=([\w-]+)&", html_page)
        if not m:
            return []
        vqd = m.group(1)
        api = f"https://duckduckgo.com/news.js?o=json&q={q}&vqd={vqd}&p=1&l=us-en&dl=en&ct=US"
        raw = urllib.request.urlopen(urllib.request.Request(api, headers=ua), context=ctx, timeout=10).read().decode("utf-8", "ignore")
        obj = json.loads(raw)
        results = []
        for it in obj.get("results", [])[:max_results]:
            ts = int(it.get("date") or 0)
            if ts >= threshold:
                results.append({
                    "title": it.get("title") or "",
                    "url": it.get("url") or "",
                    "source": it.get("source") or "",
                    "date": ts,
                    "excerpt": it.get("excerpt") or "",
                })
        return results
    except Exception:
        return []


app = Flask(__name__, template_folder="templates")
app.register_blueprint(train_bp)


@app.route("/", methods=["GET", "POST"])
def index():
    ml_confidence = None
    prediction_text = None
    sentiment_score = None
    css_class = ""
    analyzed_text = ""
    analyzed_source = ""
    recent_news = []
    bulk_results = []
    predicted_price = None

    # Fetch recent news for the right column
    for n in duckduckgo_recent_news("google stock", days=3, max_results=30):
        n["date_str"] = datetime.utcfromtimestamp(n["date"]).strftime("%Y-%m-%d %H:%M")
        recent_news.append(n)

    if request.method == "POST":
        # Bulk analyze selected news URLs
        if request.form.get("form_name") == "bulk_news":
            selected = request.form.getlist("selected_urls")
            by_url = {item["url"]: item for item in recent_news}
            for u in selected:
                txt = extract_text_from_url(u)
                if not txt:
                    continue
                comp = sia.polarity_scores(txt)["compound"]
                if comp >= 0.05:
                    direction = "UP 📈"
                elif comp <= -0.05:
                    direction = "DOWN 📉"
                else:
                    direction = "UNCERTAIN 🤔"
                title = by_url.get(u, {}).get("title", u)
                bulk_results.append({
                    "url": u,
                    "title": title,
                    "compound": comp,
                    "direction": direction,
                })
        
        action = request.form.get("action", "predict")

        if action == "reset_history":
            # Clear the stored headlines
            HEADLINE_HISTORY.clear()

        elif action == "predict":
            headline = request.form.get("headline", "")
            base_score = sia.polarity_scores(headline)["compound"]
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
        # SIMPLE, EXPLAINABLE RULE:
        # sentiment >= 0.05   -> UP
        # sentiment <= -0.05  -> DOWN
        # otherwise           -> UNCERTAIN
            if sentiment_score is not None and rf_model is not None:
              proba = rf_model.predict_proba([[sentiment_score]])
              # ML confidence (probability of UP movement)
              ml_confidence = round(float(proba[0][1]) * 100, 2)

              pred = int(proba.argmax())
              confidence = round(proba[pred] * 100, 2)



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
        # show last 10 calendar days (0 sentiment = no news)
    last_rows = (
        data[[date_col, price_col, "sentiment_mean"]]
        .tail(10)
        .copy()
    )
    last_rows = last_rows.rename(columns={date_col: "date", price_col: "close"})

    # -------- NEW: full history for interactive chart using daily_sentiment_prices.csv --------
    try:
        daily_csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "daily_sentiment_prices.csv")
        new_data = pd.read_csv(daily_csv_path)
        # Rename columns to match what index.html expects: date, close, sentiment_mean
        new_data = new_data.rename(columns={
            "published_date": "date",
            "avg_close": "close",
            "avg_sentiment": "sentiment_mean"
        })
        new_data["date"] = pd.to_datetime(new_data["date"]).dt.strftime("%Y-%m-%d")
        
        # Sort by date just in case
        new_data = new_data.sort_values("date")
        
        price_json = new_data.to_dict(orient="records")
    except Exception as e:
        print(f"Error loading daily_sentiment_prices.csv: {e}")
        # Fallback to old method if file not found or error
        price_df = data[[date_col, price_col, "sentiment_mean"]].copy()
        price_df = price_df.rename(columns={date_col: "date", price_col: "close"})
        price_df["date"] = pd.to_datetime(price_df["date"]).dt.strftime("%Y-%m-%d")
        price_json = price_df.to_dict(orient="records")
    # ---------------------------------------------------------

    return render_template(
        "index.html",
        last_rows=last_rows.to_dict(orient="records"),
        prediction=prediction_text,
        sentiment=sentiment_score,
        css_class=css_class,
        history=HEADLINE_HISTORY,
        analyzed_text=analyzed_text,
        analyzed_source=analyzed_source,
        recent_news=recent_news,
        bulk_results=bulk_results,
        selected_source_type=request.form.get("source_type", "content") if request.method == "POST" else "content",
        price_json=price_json,
        predicted_price=predicted_price, 
        ml_confidence=ml_confidence  
    )



if __name__ == "__main__":
    app.run(debug=True)
