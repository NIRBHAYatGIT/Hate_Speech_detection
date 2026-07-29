
# Updated train.py
import json
import re
import string
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "labeled_data.csv"
MODEL_PATH = BASE_DIR / "models" / "hate_speech_model.joblib"
CLASS_NAMES_PATH = BASE_DIR / "models" / "class_names.json"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"
CONF_MATRIX_PATH = BASE_DIR / "models" / "confusion_matrix.png"


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\brt\b", "", text)
    text = text.replace("#", "")
    text = re.sub(r"&(?:amp|lt|gt|quot);", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    columns = {col.lower(): col for col in df.columns}
    if "tweet" not in columns and "text" in columns:
        df.rename(columns={columns["text"]: "tweet"}, inplace=True)
    if "class" not in columns and "label" in columns:
        df.rename(columns={columns["label"]: "class"}, inplace=True)

    df = df[["tweet", "class"]].dropna()
    df["tweet"] = df["tweet"].astype(str).apply(clean_text)
    return df


def build_pipeline():
    return Pipeline([
        ("tfidf",
         TfidfVectorizer(
             stop_words="english",
             lowercase=True,
             strip_accents="unicode",
             ngram_range=(1, 2),
             min_df=2,
             max_df=0.95,
             max_features=50000,
             sublinear_tf=True,
         )),
        ("clf",
         LogisticRegression(
             solver="lbfgs",
             class_weight="balanced",
             max_iter=5000,
             random_state=42,
         ))
    ])


def save_confusion_matrix(y_true, y_pred, labels, out):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7,6))
    im = ax.imshow(cm)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    thresh = cm.max()/2 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j,i,str(cm[i,j]),ha="center",va="center",
                    color="white" if cm[i,j]>thresh else "black")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200)
    plt.close(fig)


def main():
    class_names = {0:"Hate Speech",1:"Offensive Language",2:"Neither"}

    df = load_data(DATA_PATH)
    X = df["tweet"]
    y = df["class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training distribution:")
    print(y_train.value_counts())

    model = build_pipeline()

    print("Training model...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    report = classification_report(
        y_test,
        y_pred,
        target_names=[class_names[i] for i in sorted(class_names)],
        zero_division=0,
    )

    print(f"\nAccuracy: {acc*100:.2f}%")
    print(report)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "accuracy": acc,
            "train_size": len(X_train),
            "test_size": len(X_test)
        }, f, indent=2)

    save_confusion_matrix(
        y_test,
        y_pred,
        [class_names[i] for i in sorted(class_names)],
        CONF_MATRIX_PATH,
    )

    print("\nSample Predictions:")
    samples = [
        "I love my family.",
        "Have a nice day.",
        "You are stupid.",
        "Go kill yourself.",
        "I hate black people.",
        "Thank you for your help."
    ]
    for s in samples:
        cleaned = clean_text(s)
        pred = int(model.predict([cleaned])[0])
        prob = model.predict_proba([cleaned])[0][pred]
        print(f"{s} -> {class_names[pred]} ({prob:.2%})")

    print("\nSaved:")
    print(MODEL_PATH)
    print(CLASS_NAMES_PATH)
    print(METRICS_PATH)
    print(CONF_MATRIX_PATH)


if __name__ == "__main__":
    main()
