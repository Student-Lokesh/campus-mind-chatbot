# 🎓 Campus Mind - Medicaps University Enquiry Chatbot
👉 Live Demo: Click Here to Chat with Campus Mind->(https://campus-mind-chatbot.onrender.com)

Welcome to **Campus Mind**, a smart and interactive college enquiry chatbot designed specifically for Medicaps University. This multilingual assistant helps students and parents fetch real-time information about admissions, official fee structures, courses, placements, and campus facilities.

## 📸 Project Screenshot
!Campus Mind Chatbot->(screenshot.png)

## 📄 Project Report
👉 Click here to view the complete Project Report->(Mini Project Final Report (567,543,560).pdf)

## ✨ Key Features

* 🗣️ **Bilingual Support (English & Hinglish):** Automatically detects the user's language and responds in the same tone (English or Hinglish) for a natural conversational flow.
* 💰 **Verified Fee Structures:** Provides accurate, up-to-date tuition and hostel fee details sourced directly from the official Medicaps University website.
* 🙏 **Interactive Session Closure:** Includes a dedicated gratitude-based response logic ("Thank You" functionality) for a polite and smooth end to the chat session.
* ✍️ **Smart Spell Checker:** Built-in custom vocabulary matcher that corrects common spelling mistakes automatically using `difflib`.
* 🧠 **Custom NLP Engine:** Uses a lightweight, pure-Python TF-IDF and Cosine Similarity matching engine without relying on heavy external ML libraries.
* 🌗 **Dark/Light Mode:** A sleek, fully responsive frontend UI with a built-in day/night theme toggle.

---

## 🛠️ Tech Stack

* **Backend:** Python, Flask
* **Natural Language Processing:** NLTK (Tokenization, Stemming, Stopwords)
* **Knowledge Base:** JSON (`data.json`)
* **Frontend:** HTML5, CSS3, Vanilla JavaScript (No frameworks required)

---

## 📁 Project Structure

```text
college-chatbot/
├── app.py             # Flask server & NLP intent matching logic
├── index.html         # Frontend UI (Chat interface served by Flask)
├── data.json          # Knowledge base containing intents, patterns, and responses
└── README.md          # Project Details and Steps for Setup.

🚀 How to Run the Project
Step 1: Install Dependencies
Ensure you have Python installed on your system. Run the following command to install the required libraries:

pip install -r requirements.txt

(Dependencies include flask and nltk. The app will automatically download necessary NLTK corpora like punkt and stopwords on the first run).

Step 2: Start the Flask Server
Navigate to the project directory in your terminal and start the backend server:

python app.py

You should see a startup message indicating the chatbot is trained and running on http://0.0.0.0:5000.

Step 3: Open the Interface
Open your web browser and go to:
http://127.0.0.1:5000

Click the chat bubble in the bottom right corner to start interacting with Campus Mind!

💬 Sample Interactions
You can ask the chatbot questions in English or Hinglish:

Admissions: "How to apply?" or "Admission kaise lu?"

Fees: "What is the B.Tech fee structure?" or "B.tech ki fees kitni hai?"

Hostel: "Is hostel available?" or "Hostel ka kya jugaad hai?"

Closure: "Thank you so much!" or "Thanks bhai" ---

🔧 Customization
To add new topics or update existing college information:

Open data.json.

Add a new intent block with the tag, training patterns, and language-specific responses (en and hi).

Save the file and restart app.py. The model will automatically retrain on startup.

Example:
{
  "tag": "library",
  "patterns": ["where is the library", "library kahan hai"],
  "responses": {
    "en": ["The central library is located in the main administrative block."],
    "hi": ["Central library main admin block me hai bhai."]
  }
}

Note:(Built for Medicaps University | Developed with Flask + NLTK)
