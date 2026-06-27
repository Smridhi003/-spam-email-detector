# ================================================================
# PROJECT 3: Spam Email Classifier (NLP + Machine Learning)
# Tech Stack : Python + Scikit-learn + NLP (TF-IDF)
# Author     : Smridhi | github.com/Smridhi003
#
# DATASET OPTIONS:
#   Option A (Recommended): Real Kaggle dataset
#     1. Go to: kaggle.com/datasets/uciml/sms-spam-collection-dataset
#     2. Download spam.csv → put it in this folder
#     3. Run this script — it auto-detects the file
#
#   Option B: Uses built-in sample data (no download needed)
#
# Run this FIRST → then run: streamlit run app.py
# ================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import pickle
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("     SPAM EMAIL CLASSIFIER — NLP + Machine Learning")
print("=" * 60)

# ── STEP 1: Load Dataset ─────────────────────────────────────────
import os

if os.path.exists('spam.csv'):
    # ── Real Kaggle Dataset ──────────────────────────────────────
    print("\n✅ Found spam.csv — loading real Kaggle dataset...")
    try:
        raw = pd.read_csv('spam.csv', encoding='latin-1')
        # The Kaggle SMS Spam file has columns v1 (label) and v2 (text)
        df = raw[['v1', 'v2']].copy()
        df.columns = ['label_text', 'email']
        df['label'] = (df['label_text'] == 'spam').astype(int)
        df = df[['email', 'label']].dropna()
        print(f"   Loaded {len(df)} real messages from Kaggle!")
    except Exception as e:
        print(f"   ⚠️  Could not parse spam.csv ({e}) — falling back to built-in data.")
        df = None
else:
    df = None

if df is None:
    # ── Built-in Sample Data ─────────────────────────────────────
    print("\n📝 spam.csv not found — using built-in sample data.")
    print("   Tip: Download the real dataset from kaggle.com for better results.\n")

    spam_msgs = [
        "Congratulations! You've WON a FREE iPhone! Click here NOW!",
        "URGENT: Your bank account has been compromised. Verify immediately!",
        "You're selected for a $1,000,000 lottery prize. Claim today!",
        "FREE OFFER: Lose 30 pounds in 30 days! Order now!",
        "Make money fast! Work from home — earn $5000/week guaranteed!",
        "Buy cheap meds online! No prescription needed! Lowest prices!",
        "Hot singles in your area want to meet you! Free sign up!",
        "WINNER ALERT: You've been chosen for a special cash reward!",
        "Earn extra cash just by clicking ads. Sign up FREE today!",
        "Get rich quick! Invest in crypto for 1000% returns!",
        "You've been selected for an exclusive VIP deal. Limited time!",
        "FREE GIFT CARD — Your Amazon reward is waiting. Claim now!",
        "Act now and receive 90% discount on all luxury watches!",
        "Your PayPal account is suspended. Verify your identity now.",
        "Refinance your mortgage at unbelievably low rates! Call now!",
        "Double your money in 24 hours with our secret investment plan!",
        "Claim your FREE vacation package! You have been pre-selected!",
        "ALERT: Suspicious activity detected. Click to secure account!",
        "You have a pending $500 cash reward. Click to collect now!",
        "MAKE MONEY ONLINE from home! No experience needed. $300/day!",
        "You're a winner! Reply with your credit card to receive prize!",
        "FREE trial offer. No credit card required! Sign up now!",
        "Your email was randomly selected for a special cash prize!",
        "Buy 1 get 10 FREE! This week only! Shop now!",
        "Exclusive offer: Win a brand new car! Register here FREE!",
    ] * 6

    ham_msgs = [
        "Hi, can we schedule the project meeting for tomorrow at 2pm?",
        "Please find attached the quarterly report for your review.",
        "Happy birthday! Hope you have a wonderful day today.",
        "The team dinner is confirmed for Friday at 7:30pm.",
        "I reviewed your code and left some comments on GitHub.",
        "Can you please send me the updated presentation slides?",
        "Your package has been shipped and will arrive Thursday.",
        "Are you free for a quick call this week to catch up?",
        "Reminder: your dentist appointment is scheduled for Monday.",
        "Here are the notes from today's standup meeting.",
        "Thank you for your application. We will be in touch.",
        "Your subscription receipt for this month is attached.",
        "I wanted to share this interesting article I found about ML.",
        "The library book you reserved is now available for pickup.",
        "Could you review my pull request when you get a chance?",
        "Let's catch up over coffee next week. Are you free Tuesday?",
        "I've finished the draft report — could you give it a look?",
        "The conference call is confirmed for 3pm. Joining link below.",
        "Could you please update the documentation before the deadline?",
        "Here's the invoice for last month's services. Please review.",
        "Our team has finished the sprint. Velocity was 42 points.",
        "I'm attaching the contract for your signature. No rush.",
        "The server maintenance is scheduled for this Sunday midnight.",
        "Thanks for helping me debug that issue yesterday!",
        "I've booked the conference room for our 10am meeting.",
    ] * 6

    emails = spam_msgs + ham_msgs
    labels = [1] * len(spam_msgs) + [0] * len(ham_msgs)
    df = pd.DataFrame({'email': emails, 'label': labels})
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

spam_count = df['label'].sum()
ham_count  = len(df) - spam_count
print(f"✅ Dataset ready : {len(df)} messages")
print(f"   🚫 Spam: {spam_count}   ✅ Ham: {ham_count}")

# ── STEP 2: Visualise ────────────────────────────────────────────
df['length'] = df['email'].str.len()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.bar(['Ham (Legit)', 'Spam'], [ham_count, spam_count],
        color=['#4ECDC4', '#FF6B6B'], edgecolor='white', linewidth=1.2)
ax1.set_title('Message Class Distribution', fontweight='bold')
ax1.set_ylabel('Count')
for i, v in enumerate([ham_count, spam_count]):
    ax1.text(i, v + max(ham_count, spam_count)*0.01, str(v), ha='center', fontweight='bold')

for lbl, colour, name in [(0,'#4ECDC4','Ham'), (1,'#FF6B6B','Spam')]:
    ax2.hist(df[df['label']==lbl]['length'], bins=30,
             alpha=0.7, color=colour, label=name)
ax2.set_title('Message Length Distribution', fontweight='bold')
ax2.set_xlabel('Characters'); ax2.legend()

plt.suptitle('Dataset Overview', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('data_overview.png', dpi=150, bbox_inches='tight')
print("\n✅ Saved: data_overview.png")
plt.show()

# ── STEP 3: Text Preprocessing ───────────────────────────────────
def preprocess(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', 'NUM', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean'] = df['email'].apply(preprocess)

print("\n📝 Preprocessing example:")
print(f"   Original : {df['email'].iloc[0][:65]}...")
print(f"   Cleaned  : {df['clean'].iloc[0][:65]}...")

# ── STEP 4: Top Words ────────────────────────────────────────────
def top_words(texts, n=12):
    words = ' '.join(texts).split()
    stopwords = {'the','a','an','is','it','to','in','and','of','you',
                 'for','your','this','we','be','that','have','i','NUM'}
    filtered = [w for w in words if w not in stopwords and len(w) > 2]
    return Counter(filtered).most_common(n)

spam_words = top_words(df[df['label']==1]['clean'])
ham_words  = top_words(df[df['label']==0]['clean'])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
w_s, c_s = zip(*spam_words)
ax1.barh(w_s[::-1], c_s[::-1], color='#FF6B6B', edgecolor='white')
ax1.set_title('Top Words in SPAM', fontweight='bold'); ax1.set_xlabel('Frequency')

w_h, c_h = zip(*ham_words)
ax2.barh(w_h[::-1], c_h[::-1], color='#4ECDC4', edgecolor='white')
ax2.set_title('Top Words in HAM', fontweight='bold'); ax2.set_xlabel('Frequency')

plt.suptitle('Most Frequent Words', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('word_frequency.png', dpi=150, bbox_inches='tight')
print("✅ Saved: word_frequency.png")
plt.show()

# ── STEP 5: Split ────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    df['clean'], df['label'],
    test_size=0.2, random_state=42, stratify=df['label'],
)
print(f"\n📦 Train: {len(X_train)}  |  Test: {len(X_test)}")

# ── STEP 6: Build Pipelines ──────────────────────────────────────
pipelines = {
    'Naive Bayes': Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
        ('clf',   MultinomialNB(alpha=0.1)),
    ]),
    'Logistic Regression': Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2), sublinear_tf=True)),
        ('clf',   LogisticRegression(C=10, max_iter=1000, random_state=42)),
    ]),
    'Linear SVM': Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2), sublinear_tf=True)),
        ('clf',   LinearSVC(C=1.0, max_iter=3000, random_state=42)),
    ]),
}

# ── STEP 7: Train & Evaluate ─────────────────────────────────────
results   = {}
best_f1   = 0
best_name = ''
best_pipe = None

print("\n🚀 Training Models...\n" + "-" * 60)
for name, pipe in pipelines.items():
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    r = {
        'accuracy' : accuracy_score(y_test,  y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall'   : recall_score(y_test,    y_pred),
        'f1'       : f1_score(y_test,        y_pred),
    }
    results[name] = r
    print(f"  {name}")
    print(f"    Acc:{r['accuracy']*100:.1f}%  Prec:{r['precision']*100:.1f}%"
          f"  Recall:{r['recall']*100:.1f}%  F1:{r['f1']*100:.1f}%\n")
    if r['f1'] > best_f1:
        best_f1, best_name, best_pipe = r['f1'], name, pipe

print(f"🏆 Best: {best_name}  (F1 = {best_f1*100:.1f}%)")

# ── STEP 8: Metrics Chart ────────────────────────────────────────
metric_keys = ['accuracy', 'precision', 'recall', 'f1']
names_list  = list(results.keys())
x           = np.arange(len(metric_keys))
w           = 0.25
COLORS      = ['#6C63FF', '#4ECDC4', '#FF6B6B']

fig, ax = plt.subplots(figsize=(12, 5))
for i, n in enumerate(names_list):
    vals = [results[n][m] * 100 for m in metric_keys]
    ax.bar(x + i*w, vals, w, label=n,
           color=COLORS[i], edgecolor='white', linewidth=0.8)
ax.set_title('Model Performance Comparison', fontsize=13, fontweight='bold')
ax.set_ylabel('Score (%)'); ax.set_ylim(80, 105)
ax.set_xticks(x + w); ax.set_xticklabels([m.capitalize() for m in metric_keys])
ax.legend(); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('metrics_comparison.png', dpi=150, bbox_inches='tight')
print("\n✅ Saved: metrics_comparison.png")
plt.show()

# ── STEP 9: Confusion Matrix ─────────────────────────────────────
y_pred_best = best_pipe.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)
tn, fp, fn, tp = cm.ravel()
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
            xticklabels=['Ham','Spam'], yticklabels=['Ham','Spam'],
            linewidths=0.5, linecolor='white')
plt.title(f'Confusion Matrix — {best_name}', fontsize=12, fontweight='bold')
plt.ylabel('Actual'); plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
print(f"✅ Saved: confusion_matrix.png")
print(f"   TP (spam caught)  : {tp} | FP (ham as spam): {fp}")
print(f"   FN (spam missed)  : {fn} | TN (ham correct): {tn}")
plt.show()

# ── STEP 10: Save Model ──────────────────────────────────────────
with open('spam_model.pkl', 'wb') as f:
    pickle.dump({'pipeline': best_pipe, 'name': best_name}, f)
print(f"\n✅ Model saved → spam_model.pkl")
print("🚀 Now run:  streamlit run app.py")
