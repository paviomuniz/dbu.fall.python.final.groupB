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

    def main():
        print("Loading data...")
    X, y = load_data()

    print("Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Building pipeline...")
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            ngram_range=(1, 2),
            lowercase=True
        )),
        ("clf", LinearSVC(C=1.0))
    ])

    print("Training...")
    pipeline.fit(X_train, y_train)

    print("Evaluating...")
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.4f}\n")
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=["FAKE", "REAL"]))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # ----- Save artifacts -----
    print("\nSaving artifacts...")
    tfidf = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]
    MODEL_DIR.mkdir(exist_ok=True)
    dump(tfidf, MODEL_DIR / "tfidf.joblib")
    dump(clf,   MODEL_DIR / "model.joblib")
    print("Saved models/tfidf.joblib and models/model.joblib")

    # ----- Save top global features for each class (for documentation) -----
    feature_names = np.array(tfidf.get_feature_names_out())
    coefs = np.ravel(clf.coef_)  # LinearSVC coef_ is already a dense ndarray

    # Positive weights -> REAL (class 1), Negative -> FAKE (class 0)
    top_real_idx = np.argsort(coefs)[-10:][::-1]
    top_fake_idx = np.argsort(coefs)[:10]

    doc = []
    doc.append("Top 10 REAL-weighted terms:\n")
    for f, w in zip(feature_names[top_real_idx], coefs[top_real_idx]):
        doc.append(f"{f}: {w:.4f}")

    doc.append("\nTop 10 FAKE-weighted terms:\n")
    for f, w in zip(feature_names[top_fake_idx], coefs[top_fake_idx]):
        doc.append(f"{f}: {w:.4f}")

    Path("models/top_terms.txt").write_text("\n".join(doc), encoding="utf-8")
    print("\nWrote models/top_terms.txt")


if __name__ == "__main__":
    main()
