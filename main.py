# main.py
#©Adam Basly. All rights reserved. 
#Any distribution without naming the author will be punished. 
activation=True
import subprocess
import sys
import tkinter as tk
from tkinter import messagebox
import os
if activation==False:
    sys.exit("Error-code: 0x43R43DESACTIVATED36")

# Installiere notwendige Bibliotheken
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install("nltk")
install("scikit-learn")
install("numpy")
install("requests")
install("groq")

from groq import Groq
import json
import nltk
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import os
import requests
# Lade NLTK-Daten
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker_tab')
client = Groq(api_key="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")

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, ne_chunk

from data import load_training_data, load_and_append_data

# Setze Seed für Reproduzierbarkeit
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

set_seed(42)  # Beispiel-Seed

# Lade das Trainingsdatenset aus der JSON-Datei
training_data = load_training_data()
training_data = load_and_append_data(training_data, 'comics.json', 'comicSeries')
training_data = load_and_append_data(training_data, 'dishes.json', 'dishes')
training_data = load_and_append_data(training_data, 'books.json', 'books')
training_data = load_and_append_data(training_data, 'movies.json', 'movies')
training_data = load_and_append_data(training_data, 'fruits.json', 'fruits')
training_data = load_and_append_data(training_data, 'animals.json', 'animals')
training_data = load_and_append_data(training_data, 'windows.json', 'windowsVersions')
training_data = load_and_append_data(training_data, 'deutsch6klassebayern.json', 'deutsch6klassebayern')
training_data = load_and_append_data(training_data, 'superMarioGames.json', 'superMarioGames')
training_data = load_and_append_data(training_data, 'informatik6klassebayern.json', 'informatik6klassebayern')
training_data = load_and_append_data(training_data, 'mathematik6klassebayern.json', 'mathematik6klassebayern')

if not training_data:
    raise ValueError("Das Trainingsdatenset ist leer. Bitte überprüfen Sie die Quelle der Daten.")

# Daten vorverarbeiten
vectorizer = TfidfVectorizer()
questions = [data['question'] for data in training_data]
X = vectorizer.fit_transform(questions)

answers = [data['answer'] for data in training_data]
model = SVC(kernel='linear')
model.fit(X, answers)

# Zusatzfunktionen für NLP
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
modelofAI = input("Wähle ein Modell aus (Gemma2-9b-it[1] MINT-AI[2]):")
if modelofAI == "2":
    def preprocess_text(text):
        try:
            tokens = word_tokenize(text)
            stemmed = [stemmer.stem(token) for token in tokens]
            lemmatized = [lemmatizer.lemmatize(token) for token in tokens]
            pos_tags = pos_tag(tokens)
            named_entities = ne_chunk(pos_tags)
            return {
                "tokens": tokens,
                "stemmed": stemmed,
                "lemmatized": lemmatized,
                "pos_tags": pos_tags,
                "named_entities": named_entities
            }
        except Exception as e:
            return {"error": str(e)}

    # Funktion zur Evaluierung mathematischer Ausdrücke
    def evaluate_math_expression(expression):
        try:
            result = eval(expression)
            return f"Das Ergebnis ist: {result}"
        except Exception as e:
            return f"Fehler beim Auswerten des Ausdrucks: {str(e)}"

    # Funktion zum Öffnen von VS Code
    def open_vscode():
        try:
            os.system("code")
            return "VS Code wird geöffnet."
        except Exception as e:
            return f"Fehler beim Öffnen von VS Code: {str(e)}"

    # Funktion zur Durchführung einer Websuche mit DuckDuckGo
    def search_web(query):
        try:
            url = "https://api.duckduckgo.com/"
            params = {"q": query, "format": "json"}
            response = requests.get(url, params=params)
            response.raise_for_status()
            search_results = response.json()
        
            # Überprüfen, ob Ergebnisse vorhanden sind
            if "RelatedTopics" in search_results and len(search_results["RelatedTopics"]) > 0:
                results = [f"- {topic['Text']}\n  {topic['FirstURL']}" for topic in search_results["RelatedTopics"] if "Text" in topic and "FirstURL" in topic]
                if len(results) == 0:
                    return "Keine relevanten Ergebnisse gefunden."
            
                # Formatiere die Ergebnisse
                formatted_results = "\n".join(results)
                return f"Suchergebnisse:\n{formatted_results}"
            else:
                return "Keine Ergebnisse gefunden."
        except Exception as e:
            return f"Fehler bei der Websuche: {str(e)}"

    # Funktion zur Beantwortung von Fragen
    def chatbot_response(question):
        try:
            # Überprüfen, ob die Frage ein mathematischer Ausdruck ist
            if question.startswith("Berechne"):
                expression = question.split("Berechne")[-1].strip()
                return evaluate_math_expression(expression), None
        
            # Überprüfen, ob die Frage das Öffnen von VS Code betrifft
            if question.lower() in ["öffne vs code", "code"]:
                return open_vscode(), None
        
            # Überprüfen, ob die Frage eine Websuche erfordert
            if question.lower().startswith("suche nach"):
                query = question.split("suche nach")[-1].strip()
                return search_web(query), None
        
            # Überprüfen, ob die Frage eine Begrüßung ist
            greetings = ["hallo", "hi", "hey", "guten tag"]
            if question.lower() in greetings:
                return "Hallo! Wie kann ich Ihnen helfen?", None
        
            question_tfidf = vectorizer.transform([question])
        
            response = model.predict(question_tfidf)[0]
            nlp_info = preprocess_text(question)
        
            return response, nlp_info
        except Exception as e:
            return f"Fehler bei der Verarbeitung der Frage: {str(e)}", None

    # Funktion zum Hinzufügen von neuen Fragen und Antworten
    def add_to_training_data(question, answer, file_path='training_data.json'):
        global training_data
        new_entry = {"question": question, "answer": answer}
        training_data.append(new_entry)
    
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(training_data, file, ensure_ascii=False, indent=4)
    
        # Modelle neu trainieren
        questions = [data['question'] for data in training_data]
        X = vectorizer.fit_transform(questions)
        answers = [data['answer'] for data in training_data]
        model.fit(X, answers)

    # Automatisches Lernen aus Interaktionen
    def learn_from_interaction(user_input, expected_response):
        add_to_training_data(user_input, expected_response)

    # Funktion zur Überprüfung und Verbesserung der Antwort
    def validate_response(user_input, response):
        print(f"Chatbot: {response}")
        feedback = input("War die Antwort korrekt? (ja/nein): ").strip().lower()
        if feedback == "nein":
            correct_answer = input("Wie hätte ich antworten sollen? ")
            learn_from_interaction(user_input, correct_answer)
            return correct_answer
        return response

    # Chatbot testen
    if __name__ == "__main__":
        while True:
            user_input = input("Du: ")
            if user_input.lower() in ["exit", "quit"]:
                break
        
            response, nlp_info = chatbot_response(user_input)
        
            if response is None:
                print("Ich habe deine Frage nicht verstanden. Wie sollte ich darauf antworten?")
                new_answer = input("Neue Antwort: ")
                learn_from_interaction(user_input, new_answer)
                response = new_answer
            else:
                response = validate_response(user_input, response)
        
            print("Chatbot:", response)
            print("NLP Info:", nlp_info)
elif modelofAI == "1":
    while True:
        usersinput = input("Enter your message: ")
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": usersinput,
                }
            ],
            model="llama-3.3-70b-versatile",
        )

        print(chat_completion.choices[0].message.content)
