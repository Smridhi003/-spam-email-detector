# 📧 Spam Email Detector — NLP + Machine Learning

> **NLP + ML** project that detects spam emails using text preprocessing, TF-IDF, and ML classifiers — with a **live web app** where anyone can paste a message and get an instant verdict.

![Python](https://img.shields.io/badge/Python-3.x-blue) ![scikit-learn](https://img.shields.io/badge/scikit--learn-NLP-orange) ![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red)

## 📌 What This Project Does
Classifies any email/SMS as **SPAM 🚫 or HAM ✅ (legitimate)** using Natural Language Processing. The web app lets you type or paste any message and get an instant classification.

## 🔬 How It Works
```
Raw Email Text
    ↓  preprocess (lowercase, remove URLs, punctuation, numbers)
Cleaned Text
    ↓  TF-IDF Vectorizer (words → numerical features)
Feature Matrix
    ↓  ML Classifier (Naive Bayes / Logistic Regression / SVM)
    ↓
SPAM 🚫  or  HAM ✅
```

## 📊 Upgrade with Real Data (Recommended)
This project works out of the box with built-in sample data. For better accuracy, use the real dataset:

1. Go to: [kaggle.com/datasets/uciml/sms-spam-collection-dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
2. Download `spam.csv`
3. Place it in this project folder
4. Run `python spam_classifier.py` — it auto-detects the file!

## 🛠️ Tech Stack
`Python` · `scikit-learn` · `TF-IDF` · `pandas` · `numpy` · `matplotlib` · `seaborn` · `Streamlit`

## 📁 Project Structure
```
project3-spam-classifier/
├── spam_classifier.py   ← Train models, compare, save best
├── app.py               ← Live Streamlit web app
├── spam.csv             ← (Optional) Real Kaggle dataset
├── requirements.txt     ← Dependencies
└── README.md
```

## 🚀 How to Run

```bash
# Step 1 — Install dependencies
pip install -r requirements.txt

# Step 2 — Train and save the model
python spam_classifier.py

# Step 3 — Launch the web app
streamlit run app.py
```

## 📊 Charts Generated
- `data_overview.png` — class distribution + message length analysis
- `word_frequency.png` — most common words: spam vs ham
- `metrics_comparison.png` — accuracy, precision, recall, F1 for all models
- `confusion_matrix.png` — TP / FP / TN / FN breakdown

## 🌐 Deploy for Free
1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo → **live in 2 minutes**

## 📈 Results
| Model | F1 Score |
|-------|----------|
| Naive Bayes | ~95% |
| Logistic Regression | ~97% |
| Linear SVM ✅ | ~97% |

---
**Author:** Smridhi · [GitHub](https://github.com/Smridhi003)
