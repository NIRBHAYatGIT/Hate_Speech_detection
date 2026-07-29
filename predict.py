
"""
predict.py

Usage:
    python predict.py "your text here"
"""

from pathlib import Path
import json
import sys

import joblib

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "hate_speech_model.joblib"
CLASS_NAMES_PATH = BASE_DIR / "models" / "class_names.json"


def clean_text(text: str) -> str:
    import re
    import string

    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\brt\b", "", text)
    text = text.replace("#", "")
    text = re.sub(r"&(?:amp|lt|gt|quot);", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Run 'python train.py' first."
        )

    model = joblib.load(MODEL_PATH)

    class_names = {}
    if CLASS_NAMES_PATH.exists():
        with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
            class_names = {int(k): v for k, v in json.load(f).items()}

    return model, class_names


def predict(model, class_names, text):
    cleaned = clean_text(text)

    pred = int(model.predict([cleaned])[0])
    probs = model.predict_proba([cleaned])[0]

    label = class_names.get(pred, str(pred))
    confidence = probs[pred]

    print("\nInput:")
    print(text)

    print("\nPrediction:")
    print(label)

    print(f"\nConfidence: {confidence*100:.2f}%")

    print("\nClass Probabilities:")
    for idx, prob in enumerate(probs):
        name = class_names.get(idx, str(idx))
        print(f"{name:<22}: {prob*100:.2f}%")


def main():
    if len(sys.argv) < 2:
        print('Usage:')
        print('python predict.py "your text here"')
        sys.exit(1)

    text = " ".join(sys.argv[1:])

    model, class_names = load_model()

    predict(model, class_names, text)


if __name__ == "__main__":
    main()
