import os
import logging
import threading
from typing import Optional
from flask import Flask, request, jsonify, render_template_string

# Flask app (Elastic Beanstalk Procfile expects "application:application")
application = Flask(__name__)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Resolve artifact paths relative to this file; allow env overrides (empty env won't override)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.getenv("MODEL_PATH") or os.path.join(BASE_DIR, "basic_classifier.pkl")
VECTORIZER_PATH = os.getenv("VECTORIZER_PATH") or os.path.join(BASE_DIR, "count_vectorizer.pkl")

# Log resolved paths
logger.info("CWD: %s", os.getcwd())
logger.info("Resolved MODEL_PATH: %s", MODEL_PATH)
logger.info("Resolved VECTORIZER_PATH: %s", VECTORIZER_PATH)

# Global variables for loaded artifacts
_loaded_model: Optional[object] = None
_vectorizer: Optional[object] = None
_artifact_lock = threading.Lock()

# Artifact loading
def _load_artifacts_once() -> None:
    """Lazily load model and vectorizer once per process."""
    global _loaded_model, _vectorizer
    if _loaded_model is not None and _vectorizer is not None:
        return
    with _artifact_lock:
        if _loaded_model is None or _vectorizer is None:
            import pickle
            logger.info("Loading artifacts...")
            with open(MODEL_PATH, "rb") as mf:
                _loaded_model = pickle.load(mf)
            with open(VECTORIZER_PATH, "rb") as vf:
                _vectorizer = pickle.load(vf)
            logger.info("Artifacts loaded.")

# Inference function
def _predict_text(message: str) -> str:
    """Run inference and return the predicted class as a string label."""
    _load_artifacts_once()
    X = _vectorizer.transform([message])
    pred = _loaded_model.predict(X)
    # pred[0] could be a numpy scalar; normalize to native str
    val = pred[0]
    val_py = val.item() if hasattr(val, "item") else val
    return str(val_py)

# Eager load artifacts in a background thread at startup
def _eager_load_background():
    try:
        _load_artifacts_once()
    except Exception as e:
        # Log and continue; app remains healthy and will lazy-load on first request
        logger.warning("Background eager load failed: %s", e, exc_info=True)

# Non-blocking eager load at startup
threading.Thread(target=_eager_load_background, daemon=True).start()

DEMO_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Fake News Detector — Demo</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 2rem; }
    .card { max-width: 780px; margin: 0 auto; border: 1px solid #e3e3e3; border-radius: 12px; padding: 1.25rem; box-shadow: 0 2px 8px rgba(0,0,0,.04); }
    h1 { font-size: 1.4rem; margin: 0 0 0.5rem 0; }
    textarea { width: 100%; min-height: 140px; padding: .75rem; font-size: 1rem; border: 1px solid #d0d0d0; border-radius: 8px; }
    .row { margin-top: 1rem; display: flex; gap: .75rem; align-items: center; }
    button { padding: .6rem 1rem; border: 0; border-radius: 8px; background: #2563eb; color: #fff; cursor: pointer; }
    button:disabled { background: #9aa6b2; cursor: not-allowed; }
    .pill { display: inline-block; padding: .25rem .6rem; border-radius: 999px; font-size: .85rem; }
    .ok { background: #e6f7ed; color: #096a2e; }
    .warn { background: #fff4e5; color: #8a5a00; }
    .err { background: #fdecec; color: #8a1c1c; }
    .meta { color: #666; font-size: .9rem; }
    .result { margin-top: 1rem; padding: .75rem 1rem; border-radius: 10px; background: #f8fafc; }
    code { background: #f2f2f2; padding: .14rem .35rem; border-radius: 6px; }
    footer { margin-top: 2rem; color: #777; font-size: .85rem; text-align: center; }
  </style>
</head>
<body>
  <div class="card">
    <h1>Fake News Detector — Demo</h1>
    <div class="meta">
      Model loaded:
      {% if model_loaded %}
        <span class="pill ok">Yes</span>
      {% else %}
        <span class="pill warn">Not yet</span>
      {% endif %}
      &nbsp;•&nbsp;<span>Model path:</span> <code>{{ model_path }}</code>
    </div>

    <form class="row" action="/predict-form" method="post">
      <textarea name="message" placeholder="Type or paste a news snippet here..." required></textarea>
    </form>
    <div class="row">
      <button onclick="document.forms[0].submit()">Predict</button>
      <a href="/" style="margin-left:auto;text-decoration:none">Health JSON</a>
    </div>

    {% if error %}
      <div class="result err">{{ error }}</div>
    {% endif %}
    {% if prediction %}
      <div class="result">
        <strong>Prediction:</strong>
        <span class="pill" style="background:#eef2ff;color:#1e3a8a">{{ prediction }}</span>
        <div class="meta">Raw label returned by the model.</div>
      </div>
    {% endif %}

    <div class="result">
      <div><strong>API usage:</strong> POST <code>/predict</code></div>
      <div class="meta">JSON body: <code>{"message": "your text here"}</code></div>
    </div>
  </div>

  <footer>ECE444 PRA5 — Demo page</footer>
</body>
</html>
"""

# Routes
@application.get("/")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": bool(_loaded_model is not None and _vectorizer is not None),
        "model_path": MODEL_PATH,
        "vectorizer_path": VECTORIZER_PATH,
    }), 200

# Demo page rendering endpoint
@application.get("/demo")
def demo():
    return render_template_string(
        DEMO_HTML,
        model_loaded=bool(_loaded_model is not None and _vectorizer is not None),
        model_path=MODEL_PATH,
        prediction=None,
        error=None,
    )

# Form submission endpoint for demo page
@application.post("/predict-form")
def predict_form():
    message = (request.form.get("message") or "").strip()
    if not message:
        return render_template_string(
            DEMO_HTML,
            model_loaded=bool(_loaded_model is not None and _vectorizer is not None),
            model_path=MODEL_PATH,
            prediction=None,
            error="Field 'message' is required and must be non-empty.",
        ), 400
    try:
        label = _predict_text(message)
        return render_template_string(
            DEMO_HTML,
            model_loaded=True,
            model_path=MODEL_PATH,
            prediction=label,
            error=None,
        )
    except FileNotFoundError:
        return render_template_string(
            DEMO_HTML,
            model_loaded=False,
            model_path=MODEL_PATH,
            prediction=None,
            error="Model artifacts not found on server.",
        ), 503
    except Exception as e:
        logger.exception("Inference error: %s", e)
        return render_template_string(
            DEMO_HTML,
            model_loaded=bool(_loaded_model is not None and _vectorizer is not None),
            model_path=MODEL_PATH,
            prediction=None,
            error="Inference failed.",
        ), 500

# JSON API endpoint for predictions (expects {"message": "..."} per updated handout)
@application.post("/predict")
def predict_json():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()
    if not message:
        return jsonify({"error": "Field 'message' is required and must be non-empty."}), 400
    try:
        label = _predict_text(message)
        return jsonify({"label": label}), 200
    except FileNotFoundError:
        return jsonify({"error": "Model artifacts not found on server."}), 503
    except Exception as e:
        logger.exception("Inference error: %s", e)
        return jsonify({"error": "Inference failed."}), 500

if __name__ == "__main__":
    # Local dev run; in EB, Gunicorn (from Procfile) will host the app
    port = int(os.getenv("PORT", "8000"))
    application.run(host="0.0.0.0", port=port, debug=False)
