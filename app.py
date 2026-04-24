import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

HF_REPO = "wtc8964/ai-text-data"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(HF_REPO)
    model = AutoModelForSequenceClassification.from_pretrained(HF_REPO)
    model.eval()
    return tokenizer, model
tokenizer, model = load_model()


def predict_single(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)

    score = probs[0][1].item()

    # same calibration
    score = (score - 0.5) * 0.5 + 0.5
    if len(text.split()) < 10:
        score *= 0.7

    return score


def classify_account(posts):
    scores = [predict_single(p) for p in posts if p.strip() != ""]
    avg = sum(scores) / len(scores)

    label = "Likely AI Bot" if avg >= 0.74 else "Likely Human"
    return avg, label, scores


# UI
st.title("🤖 AI Bot Detector")

st.write("Paste multiple posts (one per line):")

user_input = st.text_area("Posts")

if st.button("Analyze"):
    posts = user_input.split("\n")

    avg, label, scores = classify_account(posts)

    st.subheader(label)
    st.write(f"Confidence: {avg:.2f}")

    st.subheader("Post-level scores:")
    for i, s in enumerate(scores):
        st.write(f"Post {i+1}: {s:.2f}")