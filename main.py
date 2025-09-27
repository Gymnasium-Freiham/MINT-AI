# main.py
#©Adam Basly. All rights reserved. 
#Any distribution without naming the author will be punished. 
activation=True
import subprocess
import sys
import platform
import os
if activation==False:
    sys.exit("Error-code: 0x43R43DESACTIVATED36")
# Detect the operating system
is_windows = platform.system() == "Windows"
is_linux = platform.system() == "Linux"
no_connection = False
# Installiere notwendige Bibliotheken
if is_windows:
    def install(package, *args):
        if no_connection:
            print(f"Überspringe Installation von {package} (kein Internetzugriff erlaubt).")
            return
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, *args])
else:
    def install(package, *args):
        if no_connection:
            print(f"Überspringe Installation von {package} (kein Internetzugriff erlaubt).")
            return
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, *args, "--break-system-packages"])

install("nltk")
install("scikit-learn")
install("numpy")
install("requests")
install("groq")
install("PyQt5")
install("torch==2.0.1+cpu", "-f", "https://download.pytorch.org/whl/torch_stable.html")
install("matplotlib")

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

# Lade NLTK-Daten
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('punkt_tab')
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

# Define a PyTorch neural network for text classification
class TextClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TextClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.softmax(x)

# Initialisiere das Transformer-Modell
input_dim = len(vectorizer.vocabulary_)
hidden_dim = 128
output_dim = len(set(answers))
text_classifier = TextClassifier(input_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(text_classifier.parameters(), lr=0.001)

# Funktion zum Trainieren des PyTorch-Modells
# Ensure the model's output layer matches the number of classes before training
def train_model(X, y, epochs=10):
    global text_classifier, optimizer, criterion

    # Check if the model's output layer matches the number of classes
    num_classes = max(y) + 1
    if text_classifier.fc2.out_features != num_classes:
        text_classifier = TextClassifier(input_dim, hidden_dim, num_classes)
        optimizer = torch.optim.Adam(text_classifier.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

    text_classifier.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = text_classifier(torch.tensor(X, dtype=torch.float32))
        loss = criterion(outputs, torch.tensor(y, dtype=torch.long))
        loss.backward()
        optimizer.step()

# Trainiere das Modell mit den anfänglichen Daten
X_train = X.toarray()
y_train = [answers.index(answer) for answer in answers]
train_model(X_train, y_train)

# Zusatzfunktionen für NLP
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
modelofAI = input("Wähle ein Modell aus (Gemma2-9b-it[1] MINT-AI[2]):")

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
    def chatbot_response(question):
        print(question)
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

            # Überprüfen, ob die Frage eine IP-Geolokalisierung erfordert
            if question.lower().startswith("ip geolocation"):
                app = QApplication([])
                window = IPGeolocationApp()
                window.show()
                app.exec_()
                return "IP-Geolokalisierung gestartet.", None
        
            # Überprüfen, ob die Frage eine Begrüßung ist
            greetings = ["hallo", "hi", "hey", "guten tag"]
            if question.lower() in greetings:
                return "Hallo! Wie kann ich Ihnen helfen?", None
        
            # Preprocess the question
            question_tfidf = vectorizer.transform([question]).toarray()
            text_classifier.eval()
            with torch.no_grad():
                outputs = text_classifier(torch.tensor(question_tfidf, dtype=torch.float32))
                predicted_index = torch.argmax(outputs, dim=1).item()
                response = answers[predicted_index]

            nlp_info = preprocess_text(question)
            return response, nlp_info
        except Exception as e:
            return f"Fehler bei der Verarbeitung der Frage: {str(e)}", None

    # Ensure proper reinitialization of the model and optimizer in add_to_training_data
    def add_to_training_data(question, answer, file_path='training_data.json'):
        global training_data, text_classifier, optimizer, criterion

        new_entry = {"question": question, "answer": answer}
        training_data.append(new_entry)

        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(training_data, file, ensure_ascii=False, indent=4)

        # Update the vectorizer and labels
        questions = [data['question'] for data in training_data]
        X = vectorizer.fit_transform(questions).toarray()
        answers = [data['answer'] for data in training_data]
        y = [answers.index(answer) for answer in answers]

        # Reinitialize the model if the number of classes has changed
        new_output_dim = len(set(answers))
        if new_output_dim != text_classifier.fc2.out_features:
            text_classifier = TextClassifier(input_dim, hidden_dim, new_output_dim)
            optimizer = torch.optim.Adam(text_classifier.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()

        # Retrain the model with the updated dataset
        train_model(X, y)

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
            print("NLP Info:", nlp_info)

#elif modelofAI == "1":
#    while True:
#        usersinput = input("Enter your message: ")
#        chat_completion = client.chat.completions.create(
#            messages=[
#               {
#                    "role": "user",
#                    "content": usersinput,
#                }
#            ],
#            model="llama-3.3-70b-versatile",
#        )
#
#        print(chat_completion.choices[0].message.content)
