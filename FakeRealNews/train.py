# train.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from joblib import dump
from pathlib import Path

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

def load_data():
    # Expecting Kaggle Fake/True CSVs: Fake.csv, True.csv
    fake_path = DATA_DIR / "Fake.csv"
    true_path = DATA_DIR / "True.csv"

    fake = pd.read_csv(fake_path)
    true = pd.read_csv(true_path)

    # Many versions have a 'text' column; others have 'title' + 'text'
    def combine_text(df):
        title = df["title"].fillna("")
        text = df["text"].fillna("")
        return (title + " " + text).str.strip()

    X_fake = combine_text(fake)
    X_true = combine_text(true)

    X = pd.concat([X_fake, X_true], ignore_index=True)
    y = np.array([0]*len(X_fake) + [1]*len(X_true))  # 0=fake, 1=real

    return X, y