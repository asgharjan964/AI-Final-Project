"""
app.py  –  SpamShield Flask Backend
"""

import os, json, re, string
import joblib
from flask import Flask, render_template, request, jsonify

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder="templates", static_folder="static")

# ─── Load model artefacts ─────────────────────────────────────────────────────
MODEL_PATH   = os.path.join(BASE_DIR, "model", "spam_model.pkl")
VEC_PATH     = os.path.join(BASE_DIR, "model", "vectorizer.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "model", "metrics.json")

model      = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VEC_PATH)

with open(METRICS_PATH) as f:
    METRICS = json.load(f)

# ─── Preprocessing (mirrors train_model.py) ────────────────────────────────────
def preprocess(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " url ", text)
    text = re.sub(r"\d+", " num ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze")
def analyze():
    return render_template("analyze.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html", metrics=METRICS)

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    subject = data.get("subject", "").strip()
    body    = data.get("body", "").strip()

    if not body and not subject:
        return jsonify({"error": "Please provide email content."}), 400

    combined = preprocess(subject + " " + body)
    vec      = vectorizer.transform([combined])
    pred     = model.predict(vec)[0]
    prob     = model.predict_proba(vec)[0]

    spam_prob = float(prob[1])
    ham_prob  = float(prob[0])
    is_spam   = bool(pred == 1)

    # Human-readable confidence label
    if spam_prob >= 0.90:
        confidence_label = "Very High Confidence"
    elif spam_prob >= 0.70:
        confidence_label = "High Confidence"
    elif spam_prob >= 0.50:
        confidence_label = "Moderate Confidence"
    else:
        confidence_label = "Low Confidence"

    # Tip
    if is_spam:
        tip = ("This email shows strong indicators of spam: suspicious offers, "
               "urgent calls-to-action, or unrealistic promises. "
               "We recommend not clicking any links or replying.")
    else:
        tip = ("This email appears to be legitimate. However, always stay vigilant – "
               "review the sender address and avoid sharing sensitive information "
               "unless you are certain of the source.")

    return jsonify({
        "is_spam":          is_spam,
        "label":            "Spam" if is_spam else "Not Spam",
        "spam_probability": round(spam_prob * 100, 1),
        "ham_probability":  round(ham_prob  * 100, 1),
        "confidence_label": confidence_label,
        "tip":              tip,
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
