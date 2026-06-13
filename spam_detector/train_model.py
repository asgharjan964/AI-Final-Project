"""
train_model.py
==============
Trains the spam detection model and saves it along with evaluation metrics
and visualizations to the model/ directory.
"""

import os, json, re, string
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import joblib

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "data", "emails.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "model")
STATIC_DIR = os.path.join(BASE_DIR, "static", "images")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# ─── 1. Load Data ─────────────────────────────────────────────────────────────
print("[1/6] Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"      Total samples: {len(df)}  |  Spam: {(df.label=='spam').sum()}  |  Ham: {(df.label=='ham').sum()}")

# ─── 2. Preprocessing ─────────────────────────────────────────────────────────
print("[2/6] Preprocessing text...")

def preprocess(text: str) -> str:
    """Lowercase, remove punctuation/digits, collapse whitespace."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " url ", text)   # replace URLs
    text = re.sub(r"\d+", " num ", text)               # replace numbers
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["text"] = (df["subject"].fillna("") + " " + df["body"].fillna("")).apply(preprocess)
df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

# ─── 3. Train/Test Split ──────────────────────────────────────────────────────
print("[3/6] Splitting data (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label_num"], test_size=0.2, random_state=42, stratify=df["label_num"]
)
print(f"      Train: {len(X_train)}  |  Test: {len(X_test)}")

# ─── 4. Feature Extraction ────────────────────────────────────────────────────
print("[4/6] Extracting TF-IDF features...")
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf  = vectorizer.transform(X_test)

# ─── 5. Train Model ───────────────────────────────────────────────────────────
print("[5/6] Training Logistic Regression model...")
model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
model.fit(X_train_tfidf, y_train)

# ─── 6. Evaluate ──────────────────────────────────────────────────────────────
print("[6/6] Evaluating model...")
y_pred      = model.predict(X_test_tfidf)
y_pred_prob = model.predict_proba(X_test_tfidf)[:, 1]

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)
cm   = confusion_matrix(y_test, y_pred)

print(f"\n  Accuracy : {acc:.4f}")
print(f"  Precision: {prec:.4f}")
print(f"  Recall   : {rec:.4f}")
print(f"  F1-Score : {f1:.4f}")
print(f"\n{classification_report(y_test, y_pred, target_names=['Ham','Spam'])}")

# Save metrics as JSON (read by the Flask app for the dashboard)
metrics = {
    "accuracy":  round(acc * 100, 2),
    "precision": round(prec * 100, 2),
    "recall":    round(rec * 100, 2),
    "f1_score":  round(f1 * 100, 2),
    "confusion_matrix": cm.tolist(),
    "train_size": int(len(X_train)),
    "test_size":  int(len(X_test)),
    "total_samples": int(len(df)),
}
with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

# ─── Visualisations ───────────────────────────────────────────────────────────
# Palette matching the dark UI
BG   = "#0d1117"
CARD = "#161b22"
GRN  = "#39d353"
RED  = "#ff4c4c"
BLUE = "#58a6ff"
TEXT = "#c9d1d9"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": CARD,
    "axes.edgecolor": "#30363d", "axes.labelcolor": TEXT,
    "xtick.color": TEXT, "ytick.color": TEXT,
    "text.color": TEXT, "grid.color": "#21262d",
})

# -- Confusion Matrix --
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
            xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"],
            linewidths=1, linecolor="#30363d", ax=ax,
            annot_kws={"size": 14, "weight": "bold"})
ax.set_xlabel("Predicted Label", fontsize=11)
ax.set_ylabel("True Label", fontsize=11)
ax.set_title("Confusion Matrix", fontsize=13, pad=12, color=GRN)
plt.tight_layout()
plt.savefig(os.path.join(STATIC_DIR, "confusion_matrix.png"), dpi=120, bbox_inches="tight")
plt.close()

# -- Metrics Bar Chart --
fig, ax = plt.subplots(figsize=(6, 4))
labels  = ["Accuracy", "Precision", "Recall", "F1-Score"]
values  = [acc, prec, rec, f1]
colors  = [GRN, BLUE, "#f0883e", "#bc8cff"]
bars = ax.bar(labels, values, color=colors, width=0.5, zorder=2)
ax.set_ylim(0, 1.1)
ax.yaxis.grid(True, zorder=1)
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{val*100:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_title("Model Performance Metrics", fontsize=13, pad=12, color=GRN)
ax.set_ylabel("Score", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(STATIC_DIR, "metrics_bar.png"), dpi=120, bbox_inches="tight")
plt.close()

# -- Class Distribution --
fig, ax = plt.subplots(figsize=(4, 4))
counts = df["label"].value_counts()
wedge_props = {"linewidth": 2, "edgecolor": BG}
ax.pie(counts, labels=["Ham", "Spam"], autopct="%1.1f%%",
       colors=[GRN, RED], wedgeprops=wedge_props,
       textprops={"color": TEXT, "fontsize": 12})
ax.set_title("Class Distribution", fontsize=13, color=GRN)
plt.tight_layout()
plt.savefig(os.path.join(STATIC_DIR, "class_distribution.png"), dpi=120, bbox_inches="tight")
plt.close()

# ─── Save Artefacts ───────────────────────────────────────────────────────────
joblib.dump(model,      os.path.join(MODEL_DIR, "spam_model.pkl"))
joblib.dump(vectorizer, os.path.join(MODEL_DIR, "vectorizer.pkl"))
print("\n✓ Model and vectorizer saved to model/")
print("✓ Charts saved to static/images/")
print("✓ Metrics saved to model/metrics.json")
print("\nTraining complete!")
