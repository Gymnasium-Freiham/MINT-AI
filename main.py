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
install("PyQt5")
install("torch")
install("matplotlib")
install("simpleaudio")

from urllib.parse import quote
import simpleaudio as sa
from groq import Groq
import json
import nltk
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import os
import requests
import torch
import torch.nn as nn
from PyQt5.QtWidgets import QApplication
import os

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
    torch.manual_seed(seed)

set_seed(42)  # Beispiel-Seed

# Lade das Trainingsdatenset aus der JSON-Datei
training_data = load_training_data()
training_data = load_and_append_data(training_data, './assets/comics.json', 'comicSeries')
training_data = load_and_append_data(training_data, './assets/dishes.json', 'dishes')
training_data = load_and_append_data(training_data, './assets/books.json', 'books')
training_data = load_and_append_data(training_data, './assets/movies.json', 'movies')
training_data = load_and_append_data(training_data, './assets/fruits.json', 'fruits')
training_data = load_and_append_data(training_data, './assets/animals.json', 'animals')
training_data = load_and_append_data(training_data, './assets/windows.json', 'windowsVersions')
training_data = load_and_append_data(training_data, './assets/deutsch6klassebayern.json', 'deutsch6klassebayern')
training_data = load_and_append_data(training_data, './assets/superMarioGames.json', 'superMarioGames')
training_data = load_and_append_data(training_data, './assets/informatik6klassebayern.json', 'informatik6klassebayern')
training_data = load_and_append_data(training_data, './assets/mathematik6klassebayern.json', 'mathematik6klassebayern')

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

def send_gibberlink_sound(text, ultrasound=False):
    try:
        # Wähle den passenden Modus: audible (default) oder ultrasound
        profile = "&p=4" if ultrasound else ""
        encoded_text = quote(text)
        url = f"https://ggwave-to-file.ggerganov.com/?m={encoded_text}{profile}"
        filename = "gibberlink.wav"

        # Lade die WAV-Datei herunter
        os.system(f"curl -sS {url} --output {filename}")

        # Prüfe, ob die Datei existiert und abspielbar ist
        if os.path.exists(filename):
            try:
                subprocess.Popen(["python", "./experimental/gibberlink.py", filename])
            except Exception as audio_error:
                print(f"[Gibberlink] Audio konnte nicht abgespielt werden: {audio_error}")
        else:
            print("[Gibberlink] WAV-Datei wurde nicht erfolgreich heruntergeladen.")
    except Exception as e:
        print(f"[Gibberlink] Fehler beim Senden des Tons: {e}")
# Transformer-Architektur
class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(input_dim, model_dim)
        self.transformer = nn.Transformer(model_dim, num_heads, num_layers)
        self.fc = nn.Linear(model_dim, output_dim)

    def forward(self, src, tgt):
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        output = self.transformer(src, tgt)
        output = self.fc(output)
        return output

# Initialisiere das Transformer-Modell
input_dim = len(vectorizer.vocabulary_)
model_dim = 512
num_heads = 8
num_layers = 6
output_dim = len(set(answers))
transformer_model = TransformerModel(input_dim, model_dim, num_heads, num_layers, output_dim)

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
    import matplotlib.pyplot as plt

    def generate_math_plot(expression, filename="plot.png"):
        try:
            # Beispiel: Parabel zeichnen
            if expression == "parabel":
                x = np.linspace(-10, 10, 400)
                y = x**2
                plt.figure()
                plt.plot(x, y, label="y = x^2")
                plt.title("Parabel")
                plt.xlabel("x")
                plt.ylabel("y")
                plt.axhline(0, color='black',linewidth=0.5)
                plt.axvline(0, color='black',linewidth=0.5)
                plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
                plt.legend()
                plt.savefig(filename)
                plt.close()
                return filename
            else:
                return None
        except Exception as e:
            print(f"Fehler beim Generieren der Grafik: {e}")
            return None

    # Funktion zur Evaluierung mathematischer Ausdrücke
    def evaluate_math_expression(expression):
        try:
            # Berechne das Ergebnis
            result = eval(expression)

            # Visualisiere die Rechenschritte
            filename = visualize_calculation_steps(expression)
            if filename:
                show_plot_in_gui(filename)

            return f"Das Ergebnis ist: {result}"
        except Exception as e:
            return f"Fehler beim Auswerten des Ausdrucks: {str(e)}"
    import re

    def visualize_calculation_steps(expression, filename="calculation_steps.png"):
        try:
            # Zerlege den Ausdruck in Schritte
            steps = []
            current_expression = expression

            # Berechne Schritt für Schritt unter Berücksichtigung der Reihenfolge der Operationen
            while True:
                # Finde die innerste Klammer oder den nächsten Operator
                match = re.search(r"\(([^()]+)\)", current_expression)  # Suche nach Klammern
                if match:
                    sub_expression = match.group(1)
                    result = eval(sub_expression)
                    steps.append((f"{sub_expression} = {result}", result))
                    current_expression = current_expression.replace(f"({sub_expression})", str(result))
                else:
                    # Keine Klammern mehr, berechne den Rest
                    result = eval(current_expression)
                    steps.append((f"{current_expression} = {result}", result))
                    break

            # Erstelle die Grafik
            plt.figure(figsize=(10, 6))
            y_pos = range(len(steps))
            expressions = [step[0] for step in steps]
            results = [step[1] for step in steps]

            plt.barh(y_pos, results, color='skyblue')
            plt.yticks(y_pos, expressions)
            plt.xlabel("Ergebnisse")
            plt.title("Rechenschritte")
            plt.tight_layout()
            plt.savefig(filename)
            plt.close()
            return filename
        except Exception as e:
            print(f"Fehler bei der Visualisierung der Rechenschritte: {e}")
            return None


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
    GIBBERLINK_EXPERIMENTAL = "--gibberlink=true" in sys.argv  # oder False, je nach Bedarf

    def chatbot_response(question):
        print(question)
        try:
            # Gibberlink aktivieren, wenn EXPERIMENTAL-Modus an ist

            # Mathematischer Ausdruck
            if question.startswith("Berechne"):
                expression = question.split("Berechne")[-1].strip()
                return evaluate_math_expression(expression), None

            # VS Code öffnen
            if question.lower() in ["öffne vs code", "code"]:
                return open_vscode(), None

            # Websuche
            if question.lower().startswith("suche nach"):
                query = question.split("suche nach")[-1].strip()
                return search_web(query), None


            # Begrüßung
            greetings = ["hallo", "hi", "hey", "guten tag"]
            if question.lower() in greetings:
                return "Hallo! Wie kann ich Ihnen helfen?", None

            # NLP-Modell
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
    from PyQt5.QtWidgets import QLabel, QVBoxLayout, QDialog
    from PyQt5.QtGui import QPixmap

    def show_plot_in_gui(filename):
        dialog = QDialog()
        dialog.setWindowTitle("Mathematische Grafik")
        layout = QVBoxLayout()
        label = QLabel()
        pixmap = QPixmap(filename)
        label.setPixmap(pixmap)
        layout.addWidget(label)
        dialog.setLayout(layout)
        dialog.exec_()

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
            elif user_input.lower().startswith("zeige mir eine parabel"):
                filename = generate_math_plot("parabel")
                if filename:
                    show_plot_in_gui(filename)
                    print("Hier ist die Grafik der Parabel.")
                    continue  # Springe zur nächsten Eingabe

            response, nlp_info = chatbot_response(user_input)
        
            if response is None:
                print("Ich habe deine Frage nicht verstanden. Wie sollte ich darauf antworten?")
                new_answer = input("Neue Antwort: ")
                learn_from_interaction(user_input, new_answer)
                response = new_answer
            else:
                response = validate_response(user_input, response)
        
            print("Chatbot:", response)
            if GIBBERLINK_EXPERIMENTAL:
                send_gibberlink_sound(response)
                print("Gibberlink-Signal gesendet.", None)
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
