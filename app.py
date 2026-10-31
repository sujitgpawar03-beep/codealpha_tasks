import json
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
FAQ_PATH = Path(__file__).with_name("faqs.json")

def load_faqs():
    with FAQ_PATH.open(encoding="utf-8") as f:
        data = json.load(f)
    if not data or not all("question" in x and "answer" in x for x in data):
        raise ValueError("faqs.json must contain question and answer fields.")
    return data

FAQS = load_faqs()
QUESTIONS = [item["question"] for item in FAQS]
# Fit once at startup; the same vectorizer transforms each incoming question.
VECTORIZER = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2))
FAQ_MATRIX = VECTORIZER.fit_transform(QUESTIONS)

def find_answer(user_question):
    cleaned = user_question.strip()
    if not cleaned:
        return {"answer": "Please type a question so I can help.", "matched_question": None, "score": 0}
    query_vector = VECTORIZER.transform([cleaned])
    scores = cosine_similarity(query_vector, FAQ_MATRIX).flatten()
    best_index = int(scores.argmax())
    best_score = float(scores[best_index])
    # Avoid returning an unrelated answer when similarity is too low.
    if best_score < 0.12:
        return {"answer": "Sorry, I couldn't find a close match. Please rephrase your question or contact the support team.", "matched_question": None, "score": round(best_score, 3)}
    return {"answer": FAQS[best_index]["answer"], "matched_question": FAQS[best_index]["question"], "score": round(best_score, 3)}

@app.get("/")
def home():
    return render_template("index.html")

@app.post("/chat")
def chat():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "")[:500]
    return jsonify(find_answer(question))

@app.get("/health")
def health():
    return jsonify(status="ok", faq_count=len(FAQS))

if __name__ == "__main__":
    app.run(debug=True)
