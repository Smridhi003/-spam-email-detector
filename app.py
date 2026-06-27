# ================================================================
# PROJECT 3: Spam Detector — Streamlit Web App
# Run: streamlit run app.py
# (Make sure spam_classifier.py was run first)
# ================================================================

import streamlit as st
import pickle
import re
import string
import os

# ── Page Config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="📧 Spam Detector",
    page_icon="🚫",
    layout="centered",
)

st.title("📧 Spam Email Detector")
st.markdown("Paste any email message below and find out if it's **spam or legitimate** in one click.")
st.markdown("---")

# ── Load Model ───────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists('spam_model.pkl'):
        return None, None
    with open('spam_model.pkl', 'rb') as f:
        data = pickle.load(f)
    return data['pipeline'], data['name']

pipeline, model_name = load_model()
if pipeline is None:
    st.error("⚠️ Model not found! Please run `python spam_classifier.py` first.")
    st.stop()

st.sidebar.markdown("### ℹ️ About")
st.sidebar.markdown(f"**Model:** {model_name}")
st.sidebar.markdown("**Approach:** TF-IDF vectorization + ML classifier")
st.sidebar.markdown("**Training data:** SMS/Email messages")
st.sidebar.markdown("---")
st.sidebar.markdown("**How it works:**")
st.sidebar.markdown("1. Text is cleaned & lowercased")
st.sidebar.markdown("2. Converted to TF-IDF features")
st.sidebar.markdown("3. ML model classifies as spam/ham")

# ── Preprocess ───────────────────────────────────────────────────
def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', 'NUM', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ── Example Messages ─────────────────────────────────────────────
examples = {
    "Select an example...": "",
    "✅ Work Email": "Hi team, please find the updated project report attached. The deadline is Friday. Let me know if you have questions.",
    "✅ Personal Email": "Hey! Are you free this weekend? We're planning a small get-together at my place on Saturday evening.",
    "🚫 Spam — Prize Scam": "CONGRATULATIONS! You've been selected as today's lucky winner! Claim your FREE iPhone by clicking the link NOW! Limited time offer!",
    "🚫 Spam — Bank Scam": "URGENT NOTICE: Your bank account has been temporarily suspended due to suspicious activity. Verify your identity immediately or your account will be permanently closed.",
    "🚫 Spam — Money Scam": "Make $5000/week working from home! No experience required. Join thousands already earning! Sign up for FREE today!",
}

# ── Input Area ───────────────────────────────────────────────────
st.subheader("📝 Enter Email / Message")

selected = st.selectbox("Try a quick example:", list(examples.keys()))
default  = examples[selected]

user_input = st.text_area(
    "Or type your own message:",
    value=default,
    height=160,
    placeholder="Paste or type any email message here...",
)

col_btn, col_clear = st.columns([3, 1])
predict_clicked = col_btn.button("🔍 Analyse Message", use_container_width=True, type="primary")
if col_clear.button("🗑️ Clear", use_container_width=True):
    user_input = ""

# ── Prediction ───────────────────────────────────────────────────
if predict_clicked:
    if not user_input.strip():
        st.warning("Please enter a message to analyse.")
    else:
        clean = preprocess(user_input)
        pred  = pipeline.predict([clean])[0]

        st.markdown("---")
        st.subheader("🤖 Result")

        if pred == 1:
            st.error("## 🚫 SPAM Detected!")
            st.markdown("""
            This message shows signs of spam. Be careful of:
            - Clicking any links
            - Sharing personal or financial information
            - Responding to the sender
            """)
        else:
            st.success("## ✅ Legitimate Message (Ham)")
            st.markdown("This message appears to be safe and legitimate.")

        # Word count and character count
        word_count = len(user_input.split())
        char_count = len(user_input)
        m1, m2, m3 = st.columns(3)
        m1.metric("Words", word_count)
        m2.metric("Characters", char_count)
        m3.metric("Verdict", "SPAM 🚫" if pred == 1 else "HAM ✅")

        # Show cleaned text
        with st.expander("🔍 See how the text was preprocessed"):
            st.markdown("**Original:**")
            st.write(user_input)
            st.markdown("**After cleaning (what the model sees):**")
            st.code(clean)

# ── Spam Tips ────────────────────────────────────────────────────
st.markdown("---")
with st.expander("🛡️ Tips to Spot Spam Yourself"):
    st.markdown("""
    - **ALL CAPS** and excessive **!!!** are red flags
    - Promises of **FREE prizes**, **lottery wins**, or **cash rewards**
    - **Urgency** — "Act NOW!", "Limited time!", "Expires today!"
    - Requests for **personal info** or **bank details**
    - **Suspicious links** — hover before clicking
    - **Too good to be true** offers
    """)

st.markdown("---")
st.caption("Built by **Smridhi** · [GitHub](https://github.com/Smridhi003)")
