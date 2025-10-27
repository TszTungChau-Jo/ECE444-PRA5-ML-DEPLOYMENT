# application.py
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import os

def load_model():
    """Load vectorizer and classifier from local pickle files."""
    with open("count_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("basic_classifier.pkl", "rb") as f:
        classifier = pickle.load(f)
    return vectorizer, classifier

vectorizer, classifier = load_model()

# Beanstalk looks for a module-level variable named `application`
application = Flask(__name__)

@application.get("/")
def root():
    return jsonify(
        status="ok",
        message="Fake-news detector API",
        endpoints=["/predict (POST)"]
    )

@application.get("/health")
def health():
    return jsonify(healthy=True)

@application.post("/predict")
def predict():
    """
    JSON body: {"text": "<news snippet>"}
    Returns 1 for FAKE, 0 for REAL (strings are also echoed).
    """
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")

    if not isinstance(text, str) or not text.strip():
        return jsonify(error="Provide a non-empty 'text' string in JSON body."), 400

    X = vectorizer.transform([text])

    # Model may return strings like "FAKE"/"REAL"
    raw = classifier.predict(X)[0]               # e.g., "FAKE", "REAL", 1, 0, etc.
    raw_str = str(raw).strip().lower()           # handle numpy.str_ etc.

    # Normalize to numeric: FAKE -> 1, REAL -> 0
    label_map = {"fake": 1, "real": 0}
    if raw_str in label_map:
        y = label_map[raw_str]
    else:
        # If the model already returns numeric labels, keep them
        try:
            y = int(raw)
        except Exception:
            # Fallback if label is unexpected
            y = 1 if "fake" in raw_str else 0

    # Optional: probability if available; try to pick prob of "fake"
    proba = None
    try:
        probs = classifier.predict_proba(X)[0]
        # If classifier.classes_ are strings like ['FAKE','REAL'], find index for 'fake'
        if hasattr(classifier, "classes_"):
            classes = [str(c).strip().lower() for c in classifier.classes_]
            idx_fake = classes.index("fake") if "fake" in classes else 1  # default to positive class at 1
            proba = float(probs[idx_fake])
        else:
            proba = float(probs[1])  # common convention: class 1 is positive
    except Exception:
        pass

    return jsonify(
        prediction=y,                    # 1 or 0
        label="fake" if y == 1 else "real",
        raw_label=str(raw),              # echo original model label to help debugging
        proba=proba
    )


if __name__ == "__main__":
    # Handy for local testing; Beanstalk will use WSGI
    port = int(os.environ.get("PORT", 5000))
    application.run(host="0.0.0.0", port=port)
