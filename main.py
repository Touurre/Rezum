from transformers import pipeline
from keybert import KeyBERT
import nltk
from nltk import sent_tokenize
import json

# Téléchargement des ressources NLTK
nltk.download('punkt')

# Chargement des modèles
print("Loading summarization model...")
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)

print("Loading keyword model...")
kw_model = KeyBERT("all-MiniLM-L6-v2")

# Fonctions NLP
def extract_keywords(text, top_n=8):
    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 3),
        stop_words="english",
        use_maxsum=True,
        top_n=top_n
    )
    return [kw[0] for kw in keywords]

def extract_questions(text):
    sentences = sent_tokenize(text)
    return [s for s in sentences if s.strip().endswith('?')]

def generate_json_summary(text):
    try:
        summary = summarizer(text, max_length=50, min_length=30, do_sample=False)[0]['summary_text']
        keywords = extract_keywords(text)
        questions = extract_questions(text)

        return {
            "summary": summary,
            "keywords": keywords,
            "questions": questions
        }
    except Exception as e:
        return {
            "error": str(e)
        }

# Interface console
print("System ready! Paste your conversation (type 'quit' to exit)\n")

while True:
    try:
        user_input = input(">>> Conversation: ")

        if user_input.lower() in ('quit', 'exit', 'q'):
            print("\nGoodbye!")
            break

        if len(user_input.strip()) < 40:
            print("Please enter at least 40 characters")
            continue

        result = generate_json_summary(user_input)
        print(json.dumps(result, indent=2))

    except KeyboardInterrupt:
        print("\nSession ended.")
        break
    except Exception as e:
        print(f"Error: {e}")
