"""
Medicaps University - College Enquiry Chatbot
Flask Backend with NLTK-based TF-IDF Intent Matching
"""

import difflib
import json
import random
import re
import os
import string
from flask import Flask, request, jsonify, render_template_string

# ── NLTK bootstrap ──────────────────────────────────────────────────────────
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer

    # Download required NLTK assets on first run
    for pkg in ["punkt", "stopwords", "punkt_tab"]:
        try:
            nltk.data.find(f"tokenizers/{pkg}" if "punkt" in pkg else f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)

    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("[WARN] NLTK not installed – falling back to basic keyword matching.")

# ── TF-IDF / Cosine similarity (no sklearn required) ─────────────────────────
import math
from collections import Counter


# ─────────────────────────────────────────────────────────────────────────────
# Text Preprocessing
# ─────────────────────────────────────────────────────────────────────────────
stemmer = PorterStemmer() if NLTK_AVAILABLE else None
STOP_WORDS = set()
if NLTK_AVAILABLE:
    try:
        STOP_WORDS = set(stopwords.words("english"))
    except Exception:
        pass

# --- NAYA LOGIC: Hinglish Stopwords Add Karo ---
hinglish_stops = {
    "hai", "hain", "hu", "ho", "tha", "thi", 
    "tu", "tum", "mera", "meri", "mere", "iska", "uska", 
    "kya", "kaise", "kab", "kyu",
    "ka", "ki", "ke", "ko", "se", "pe", "ye", "wo", "hi",
    "ladka", "bc"
}
STOP_WORDS.update(hinglish_stops)


def preprocess(text: str) -> list[str]:
    """Lowercase → tokenize → remove punctuation/stopwords → stem."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)

    if NLTK_AVAILABLE:
        tokens = word_tokenize(text)
    else:
        tokens = text.split()

    tokens = [t for t in tokens if t not in STOP_WORDS and t not in string.punctuation and len(t) > 1]

    if stemmer:
        tokens = [stemmer.stem(t) for t in tokens]

    return tokens


# ─────────────────────────────────────────────────────────────────────────────
# Tiny TF-IDF engine (pure Python, no dependencies)
# ─────────────────────────────────────────────────────────────────────────────
class TFIDFMatcher:
    def __init__(self):
        self.documents: list[list[str]] = []   # tokenised pattern docs
        self.tags: list[str] = []               # intent tag per doc
        self.idf: dict[str, float] = {}

    def _tf(self, tokens: list[str]) -> dict[str, float]:
        count = Counter(tokens)
        total = max(len(tokens), 1)
        return {w: c / total for w, c in count.items()}

    def _cosine(self, vec_a: dict, vec_b: dict) -> float:
        keys = set(vec_a) & set(vec_b)
        if not keys:
            return 0.0
        dot = sum(vec_a[k] * vec_b[k] for k in keys)
        mag_a = math.sqrt(sum(v * v for v in vec_a.values()))
        mag_b = math.sqrt(sum(v * v for v in vec_b.values()))
        return dot / (mag_a * mag_b + 1e-9)

    def fit(self, documents: list[list[str]], tags: list[str]):
        self.documents = documents
        self.tags = tags
        N = len(documents)
        df: dict[str, int] = {}
        for doc in documents:
            for word in set(doc):
                df[word] = df.get(word, 0) + 1
        self.idf = {w: math.log((N + 1) / (freq + 1)) + 1 for w, freq in df.items()}

    def _tfidf_vec(self, tokens: list[str]) -> dict[str, float]:
        tf = self._tf(tokens)
        return {w: tf[w] * self.idf.get(w, 1.0) for w in tf}

    def predict(self, tokens: list[str], threshold: float = 0.15) -> str | None:
        if not tokens:
            return None
        query_vec = self._tfidf_vec(tokens)
        best_score, best_tag = 0.0, None
        for doc_tokens, tag in zip(self.documents, self.tags):
            doc_vec = self._tfidf_vec(doc_tokens)
            score = self._cosine(query_vec, doc_vec)
            if score > best_score:
                best_score, best_tag = score, tag
        return best_tag if best_score >= threshold else None


# ─────────────────────────────────────────────────────────────────────────────
# Load intents & train matcher
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data.json")

with open(DATA_FILE, "r", encoding="utf-8") as f:
    intent_data = json.load(f)

intents: dict[str, dict] = {intent["tag"]: intent for intent in intent_data["intents"]}

all_docs: list[list[str]] = []
all_tags: list[str] = []

for intent in intent_data["intents"]:
    if intent["tag"] == "unknown":
        continue
    for pattern in intent["patterns"]:
        tokens = preprocess(pattern)
        if tokens:
            all_docs.append(tokens)
            all_tags.append(intent["tag"])

matcher = TFIDFMatcher()
matcher.fit(all_docs, all_tags)
print(f"[INFO] Chatbot trained on {len(all_docs)} patterns across {len(intents)-1} intents.")


# ─────────────────────────────────────────────────────────────────────────────
# Flask Application
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)


# --- Naya Spell Checker Logic ---
def build_vocab(intents_data):
    """JSON patterns se saare valid words ki list banata hai"""
    vocab = set()
    for intent in intents_data.get("intents", []):
        for pattern in intent.get("patterns", []):
            for word in pattern.lower().split():
                vocab.add(word)
    # Kuch common words manually add kar do taaki wo galat correct na hon
    vocab.update({'bhai', 'kya', 'hai', 'kaise', 'kab', 'medicaps', 'musat', 'yrr', 'bro'})
    return list(vocab)

# Ek baar start hone pe vocabulary load kar lo
VOCABULARY = build_vocab(intents)

def correct_spelling(text):
    """User ke text ko vocabulary ke hisaab se theek karta hai"""
    corrected_words = []
    for word in text.lower().split():
        # difflib check karega ki input word hamari vocab ke kis word se 75% se zyada match karta hai
        matches = difflib.get_close_matches(word, VOCABULARY, n=1, cutoff=0.75)
        if matches:
            corrected_words.append(matches[0])  # Sahi spelling use karo
        else:
            corrected_words.append(word)        # Match nahi mila toh jaisa hai waisa rehne do
    return " ".join(corrected_words)

# --- Naya Language Detector Function ---
def detect_language(text):
    """Detects if the input is Hinglish based on common keywords."""
    # Common Hinglish words ka set
    hinglish_keywords = {'kya', 'hai', 'kaise', 'kab', 'kitna', 'kitni', 'bhai', 'mujhe', 'mera', 'ka', 'ki', 'ke', 'ho', 'ha', 'nahi', 'lu', 'bhare', 'wala', 'konsi', 'kahan', 'yrr', 'chal', 'bata'}
    
    # Text ko lowercase karke punctuation hatao
    clean_text = text.lower().replace('?', '').replace('.', '').replace(',', '')
    words = clean_text.split()
    
    # Agar koi bhi word match hota hai, toh Hinglish ('hi')
    for word in words:
        if word in hinglish_keywords:
            return "hi"
    
    return "en" # Default English

def correct_spelling(text):
    """User ke text ko vocabulary ke hisaab se theek karta hai"""
    corrected_words = []
    for word in text.lower().split():
        # difflib check karega ki input word hamari vocab ke kis word se 75% se zyada match karta hai
        matches = difflib.get_close_matches(word, VOCABULARY, n=1, cutoff=0.75)
        if matches:
            corrected_words.append(matches[0])  # Sahi spelling use karo
        else:
            corrected_words.append(word)        # Match nahi mila toh jaisa hai waisa rehne do
    return " ".join(corrected_words)
# --- Update kiya gaya get_response Function ---
def get_response(user_message):
    """Generates the final reply matching the user's language."""
    tokens = preprocess(user_message)
    tag = matcher.predict(tokens)
    
    # User ki language detect karo
    lang = detect_language(user_message)

    if tag and tag in intents:
        intent_data = intents[tag]
    else:
        intent_data = intents["unknown"]

    # Check karo ki JSON me responses dict (language wise) hai ya purani list hai
    if isinstance(intent_data["responses"], dict):
        # Jo language detect hui uska array uthao, agar na mile toh 'en' uthao
        responses = intent_data["responses"].get(lang, intent_data["responses"]["en"])
    else:
        # Fallback agar purana list format ho
        responses = intent_data["responses"]

    return random.choice(responses)


@app.route("/")
def home():
    """Serve the frontend HTML from file."""
    html_path = os.path.join(BASE_DIR, "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


@app.route("/chat", methods=["POST"])
def chat():
    """REST endpoint: receives JSON {message: str} → returns {reply: str}."""
    data = request.get_json(force=True, silent=True) or {}
    user_message = str(data.get("message", "")).strip()

    if not user_message:
        return jsonify({"reply": "Please type a message!", "status": "error"}), 400

    reply = get_response(user_message)
    return jsonify({"reply": reply, "status": "ok"})


@app.route("/intents", methods=["GET"])
def list_intents():
    """Debug endpoint – lists all available intent tags."""
    return jsonify({"intents": list(intents.keys())})


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*55)
    print("  🎓 Medicaps University Chatbot – Starting Up")
    print("  Visit: http://127.0.0.1:5000")
    print("="*55 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
