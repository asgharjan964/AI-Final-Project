# ⚡ SpamShield — AI Email Spam Detection

> **Sukkur IBA University** | Department of Computer Science | AI Semester Project | Spring 2026

A full-stack machine learning web application that classifies emails as **Spam** or **Not Spam** in real time using TF-IDF feature extraction and Logistic Regression.

---

## 🚀 Quick Start

### 1. Prerequisites
- Python **3.10+**
- pip

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model *(run once)*

```bash
python train_model.py
```

This will:
- Load and preprocess `data/emails.csv`
- Train and evaluate the Logistic Regression model
- Save `model/spam_model.pkl` and `model/vectorizer.pkl`
- Save evaluation charts to `static/images/`
- Print accuracy, precision, recall, and F1-score

### 4. Start the web application

```bash
python app.py
```

Open your browser at: **http://localhost:5000**

---

## 📁 Project Structure

```
spam_detector/
├── app.py                  # Flask backend + REST API
├── train_model.py          # ML pipeline (preprocessing → training → evaluation)
├── requirements.txt        # Python dependencies
├── README.md               # This file
│
├── data/
│   └── emails.csv          # Dataset (2000 labelled emails)
│
├── model/
│   ├── spam_model.pkl      # Trained Logistic Regression model
│   ├── vectorizer.pkl      # Fitted TF-IDF vectorizer
│   └── metrics.json        # Evaluation metrics (read by dashboard)
│
├── templates/
│   ├── base.html           # Shared layout (nav + footer)
│   ├── index.html          # Home page
│   ├── analyze.html        # Email analyzer page
│   ├── dashboard.html      # Model metrics dashboard
│   └── about.html          # Project info page
│
└── static/
    ├── css/main.css        # Full stylesheet
    ├── js/
    │   ├── main.js         # Global JS (nav toggle)
    │   └── analyze.js      # Analyzer logic (API call + result rendering)
    └── images/             # Generated evaluation charts (PNG)
        ├── confusion_matrix.png
        ├── metrics_bar.png
        └── class_distribution.png
```

---

## 🧠 ML Pipeline

| Step | Details |
|------|---------|
| **Dataset** | 2,000 emails — 1,000 spam / 1,000 ham |
| **Preprocessing** | Lowercase, URL normalisation, digit tokens, punctuation removal |
| **Feature extraction** | TF-IDF vectoriser, unigram + bigram, 5,000 features, sublinear TF |
| **Train/Test split** | 80% train / 20% test (stratified) |
| **Algorithm** | Logistic Regression (C=1.0, L2, max_iter=1000) |
| **Serialisation** | Joblib (model + vectorizer saved as `.pkl`) |

### Evaluation Metrics

| Metric | Score |
|--------|-------|
| Accuracy  | 100% |
| Precision | 100% |
| Recall    | 100% |
| F1-Score  | 100% |

---

## 🌐 Web Pages

| Route | Description |
|-------|-------------|
| `/`          | Home page — project intro, stats, feature cards |
| `/analyze`   | Email analyzer — paste email, get instant verdict |
| `/dashboard` | Model dashboard — all metrics, charts, confusion matrix |
| `/about`     | About page — technical details, pipeline, challenges |
| `/api/predict` | REST endpoint (POST JSON `{subject, body}`) |

---

## 📡 API Reference

**POST** `/api/predict`

**Request body:**
```json
{
  "subject": "Congratulations! You won a prize",
  "body": "Click here to claim your reward immediately..."
}
```

**Response:**
```json
{
  "is_spam": true,
  "label": "Spam",
  "spam_probability": 98.7,
  "ham_probability": 1.3,
  "confidence_label": "Very High Confidence",
  "tip": "This email shows strong indicators of spam..."
}
```

---

## 📚 AI Concepts Applied

- **Supervised Classification** — Logistic Regression
- **Text Preprocessing** — normalisation pipeline
- **Feature Engineering** — TF-IDF with n-gram range
- **Model Evaluation** — Accuracy, Precision, Recall, F1, Confusion Matrix
- **Model Persistence** — Joblib serialisation
- **Web Deployment** — Flask REST API + responsive HTML/CSS/JS frontend

---

## 👩‍🏫 Submission Details

- **Course:** Artificial Intelligence
- **Institution:** Sukkur IBA University
- **Semester:** Spring 2026
- **Deliverable:** Implementation + GUI (4 Marks) | Report (1 Mark) | Presentation (4 Marks)
