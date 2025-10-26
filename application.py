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
    Returns: {"prediction": 0|1, "label": "real|fake", "proba": 0.0-1.0 (if available)}
    """
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")

    if not isinstance(text, str) or not text.strip():
        return jsonify(error="Provide a non-empty 'text' string in JSON body."), 400

    X = vectorizer.transform([text])
    y = int(classifier.predict(X)[0])

    # Try to include probability if the model supports it
    proba = None
    try:
        proba = float(classifier.predict_proba(X)[0][1])  # prob of class 1 (fake)
    except Exception:
        pass

    return jsonify(
        prediction=y,
        label="fake" if y == 1 else "real",
        proba=proba
    )

if __name__ == "__main__":
    # Handy for local testing; Beanstalk will use WSGI
    port = int(os.environ.get("PORT", 5000))
    application.run(host="0.0.0.0", port=port)
