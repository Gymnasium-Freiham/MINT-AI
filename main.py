# main.py
#©Adam Basly. All rights reserved. 
#Any distribution without naming the author will be punished. 
activation=True
import subprocess
import sys
import tkinter as tk
from tkinter import messagebox
import os
import unicodedata
if activation==False:
    sys.exit("Error-code: 0x43R43DESACTIV7ATED36")

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

from data import load_training_data, load_and_append_data, fetch_wikipedia_summary, fetch_wikipedia_variants, fetch_wikipedia_page_text, fetch_wiktionary_definition, extract_subject_from_question

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
# training_data = load_and_append_data(training_data, './assets/githubrepos.json', 'githubrepos')

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
modelofAI = input("Wähle ein Modell aus (Gemma2-9b-it[1] LATIN-AI[2]):")

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


    # --- NEW: extract measurement snippets from text and try wiki variants ---
    def extract_measurement_from_text(text):
        if not text:
            return None
        # normalize whitespace
        txt = re.sub(r'\s+', ' ', text)
        # common unit patterns (DE + EN) and numeric ranges
        patterns = [
            r'(\d{1,4}(?:[.,]\d+)?\s?(?:m|Meter|Metern|meter|metres|meters|cm|Zentimeter|centimeter|km|Millimeter|mm|ft|feet|in|inch|Zoll))',
            r'(\d{1,4}(?:[.,]\d+)?\s?(?:cm|Zentimeter|centimeter))',
            r'(\d{1,4}(?:[.,]\d+)?\s?(?:kg|Kilogramm|g|Gramm|lbs|pounds))',
            r'(\d+(?:[.,]\d+)?\s?[-–]\s?\d+(?:[.,]\d+)?\s?(?:m|cm|ft|in|mm))',
            r'(bis\s+zu\s+\d+(?:[.,]\d+)?\s?(?:m|cm|kg))',
            r'(etwa\s+\d+(?:[.,]\d+)?\s?(?:m|cm|kg))',
            r'(~\s?\d+(?:[.,]\d+)?\s?(?:m|cm|kg))',
        ]
        for p in patterns:
            m = re.search(p, txt, flags=re.IGNORECASE)
            if m:
                return m.group(1).strip()
        # phrases like "average/typical length is 4.5 m" (EN/DE)
        m = re.search(r'(?:average|typical|mean|durchschnittlich|im Durchschnitt)\s+(?:length|Länge)\s*(?:is|ist|:)?\s*([\d.,]+\s?(?:m|cm|metres|meters|ft|feet|in|inch|Zentimeter|Zoll))', txt, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()
        # "length is X m" or "ist X m"
        m = re.search(r'(?:length|Länge)\s*(?:is|ist|:)?\s*([\d.,]+\s?(?:m|cm|ft|in|metres|meters|Zentimeter|Zoll))', txt, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()
        # fallback: "ist X lang" style
        m = re.search(r'ist\s+(?:etwa|ungefähr|ca\.?|circa)?\s*([\d.,\s\-–]+)\s?(m|cm|kg|Meter|Zentimeter|Kilogramm|Zoll|inch|feet)', txt, flags=re.IGNORECASE)
        if m:
            num = m.group(1).strip().replace(' ', '')
            unit = m.group(2).strip()
            return f"{num}{unit}"
        return None

    # --- NEW: normalize subject and map common synonyms ---
    def normalize_subject(subj):
        if not subj:
            return None
        # remove diacritics, collapse whitespace, lowercase
        s = unicodedata.normalize('NFKD', subj)
        s = ''.join(ch for ch in s if not unicodedata.combining(ch))
        s = re.sub(r'\s+', ' ', s).strip()
        return s

    # --- REPLACED/EXTENDED: try targeted queries and DuckDuckGo + english variants ---
    def find_measurement_for_subject(subj):
        """
        Try multiple queries (wiki variants + targeted 'Länge' queries + DuckDuckGo snippets)
        Return tuple (measurement_string, source_text_short) or (None, None).
        """
        if not subj:
            return None, None

        subj_norm = normalize_subject(subj)
        # build candidate queries (DE + EN) and synonyms
        candidates = []
        # base
        candidates.append(subj)
        candidates.append(subj_norm)
        # German targeted forms
        candidates += [
            f"{subj} Länge",
            f"Länge {subj}",
            f"Durchschnittliche Länge {subj}",
            f"Durchschnittliche Länge {subj_norm}",
            f"{subj} Größe",
        ]
        # synonyms for common terms
        if re.search(r'\bauto\b', subj_norm, flags=re.IGNORECASE) or re.search(r'\bwagen\b', subj_norm, flags=re.IGNORECASE):
            candidates += ["Auto", "Personenkraftwagen", "PKW", "Durchschnittliche Länge Auto", "Durchschnittliche Länge PKW"]
        if re.search(r'\brüssel\b', subj_norm, flags=re.IGNORECASE):
            candidates += ["Rüssel", "Elefantenrüssel", "Rüssel Elefant Länge"]

        # english fallbacks
        candidates += [
            f"{subj_norm} length",
            "car length",
            "average car length",
            "typical car length",
            "elephant trunk length",
            "length of elephant trunk"
        ]

        tried = set()
        # try Wikipedia page extracts first (better chance to contain numbers)
        for q in candidates:
            if not q or q in tried:
                continue
            tried.add(q)
            try:
                page_text = fetch_wikipedia_page_text(q)
            except Exception:
                page_text = None
            if page_text:
                measurement = extract_measurement_from_text(page_text)
                if measurement:
                    return measurement, f"Wikipedia page: {q}"
            # fallback to summary if page_text didn't exist
            try:
                wiki = fetch_wikipedia_summary(q)
            except Exception:
                wiki = None
            if wiki:
                measurement = extract_measurement_from_text(wiki)
                if measurement:
                    return measurement, f"Wikipedia summary: {q}"

        # also try broader Wikipedia variants (existing helper)
        try:
            wiki_variant = fetch_wikipedia_variants(subj)
        except Exception:
            wiki_variant = None
        if wiki_variant:
            measurement = extract_measurement_from_text(wiki_variant)
            if measurement:
                return measurement, "Wikipedia (variant)"

        # lastly try DuckDuckGo search snippets for a few prioritized queries
        for q in [f"{subj} Länge", f"{subj_norm} length", "durchschnittliche Länge Auto", "average car length", "elephant trunk length"]:
            try:
                sd = search_web(q)
            except Exception:
                sd = None
            if sd and isinstance(sd, str):
                measurement = extract_measurement_from_text(sd)
                if measurement:
                    src = "DuckDuckGo: " + q
                    return measurement, src

        return None, None

    # Funktion zur Beantwortung von Fragen
    GIBBERLINK_EXPERIMENTAL = "--gibberlink=true" in sys.argv  # oder False, je nach Bedarf

    def chatbot_response(question):
        print(question)
        try:
            # quick math detection: evaluate simple arithmetic expressions even without "Berechne"
            # matches inputs containing digits and math operators (e.g. "2*4", "12 / (3+1)")
            if re.search(r'\d', question) and re.search(r'[\+\-\*\/\^()]', question):
                try:
                    # use existing evaluate_math_expression when available
                    result = evaluate_math_expression(question)
                    return result, {"tokens": [], "note": "evaluated-as-math"}
                except Exception:
                    # fallback: continue to NLP if evaluation fails
                    pass

            # Gibberlink aktivieren, wenn EXPERIMENTAL-Modus an ist

            # NEW: Definition/Übersetzungsanfragen (Wortbedeutung)
            if re.search(r'\bwas\s+bedeutet\b|\bwas\s+heißt\b|\bbedeutung\s+von\b', question, flags=re.IGNORECASE):
                term = extract_subject_from_question(question)
                if term:
                    # prefer wiktionary, fallback to wikipedia summary/page
                    definition = fetch_wiktionary_definition(term)
                    if not definition:
                        # try wikipedia page text then summary
                        definition = fetch_wikipedia_page_text(term) or fetch_wikipedia_summary(term)
                    if definition:
                        first_para = definition.split("\n\n")[0].strip()
                        return f"'{term}' bedeutet:\n{first_para}", {"tokens": [], "note": "dictionary"}
                return "Dazu konnte ich leider keine Definition finden.", {"tokens": [], "note": "no-definition"}

            # REPLACED: improved measurement question handling
            if re.search(r'\bwie\s+(lang|groß|hoch|schwer|alt)\b', question, flags=re.IGNORECASE):
                subj = extract_subject_from_question(question)
                # first try direct wiki variants (keeps previous behavior)
                wiki = fetch_wikipedia_variants(subj) if subj else None
                if wiki:
                    measurement = extract_measurement_from_text(wiki)
                    if measurement:
                        subj_display = subj or "das Objekt"
                        return f"Der {subj_display} ist ungefähr {measurement}. (Quelle: Wikipedia)", {"tokens": [], "note": "wikipedia-measurement"}
                # if initial wiki summary didn't contain measurement, do targeted search attempts
                measurement, source = find_measurement_for_subject(subj)
                if measurement:
                    subj_display = subj or "das Objekt"
                    return f"Der {subj_display} ist ungefähr {measurement}. (Quelle: {source})", {"tokens": [], "note": "web-measurement"}
                # fallback: if we had a wiki without measurement, return it (better than nothing)
                if wiki:
                    subj_display = subj or "das Objekt"
                    return f"Ich habe zu '{subj_display}' folgende Info auf Wikipedia gefunden:\n{wiki}", {"tokens": [], "note": "wikipedia-summary"}
                # nothing found -> fall back to general NLP below (or inform about missing data)
                return "Dazu habe ich leider keine eindeutige Längenangabe gefunden.", {"tokens": [], "note": "no-measurement"}

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

            # Try to enrich the model response with a deterministic Wikipedia summary
            try:
                subj = extract_subject_from_question(question)
                # do not attempt wiki if subject extraction failed (or was math)
                wiki = fetch_wikipedia_summary(subj) if subj else None
                if wiki:
                    # append but avoid duplicating if already included
                    if isinstance(response, str) and "[Wikipedia]" not in response:
                        response = (response or "") + "\n\n[Wikipedia]: " + wiki
            except Exception:
                # degrade gracefully: return model response without augmentation
                pass

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
