# data.py
import json

def load_training_data(file_path='./assets/training_data.json'):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def load_json_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def append_data(training_data, new_data, key):
    if key == "comicSeries":
        for entry in new_data["comicSeries"]:
            question = f"Was ist die Comic-Serie '{entry['title']}'?"
            answer = entry['description']
            training_data.append({"question": question, "answer": answer})
    elif key == "githubrepos":
        for entry in new_data["githubrepos"]:
            question = f"Welche URL hat '{entry['names']}'?"
            answer = entry['url']
            training_data.append({"question": question, "answer": answer})
    elif key == "dishes":
        for entry in new_data["dishes"]:
            question = f"Was ist das Gericht '{entry['name']}'?"
            answer = entry['description']
            training_data.append({"question": question, "answer": answer})
    elif key == "books":
        for entry in new_data["books"]:
            question = f"Was ist das Buch '{entry['title']}'?"
            answer = entry['description']
            training_data.append({"question": question, "answer": answer})
    elif key == "movies":
        for entry in new_data["movies"]:
            question = f"Was ist der Film '{entry['title']}'?"
            answer = entry['description']
            training_data.append({"question": question, "answer": answer})
    elif key == "fruits":
        for entry in new_data["fruits"]:
            question = f"Was ist die Frucht '{entry['name']}'?"
            answer = entry['description']
            training_data.append({"question": question, "answer": answer})
    elif key == "animals":
        for entry in new_data["animals"]:
            question = f"Was ist das Tier '{entry['name']}'?"
            answer = entry['description']
            training_data.append({"question": question, "answer": answer})
    elif key == "windowsVersions":
        for entry in new_data["windowsVersions"]:
            question = f"Was ist Windows {entry['version']}?"
            answer = entry['description']
            training_data.append({"question": question, "answer": answer})
    elif key == "deutsch6klassebayern":
        for entry in new_data["deutsch6klassebayern"]:
            question = f"Was ist das Thema '{entry['topic']}'?"
            answer = entry['description']
            training_data.append({"question": question, "answer": answer})
    elif key == "superMarioGames":
        for entry in new_data["superMarioGames"]:
            question = f"Was ist das Super Mario Spiel '{entry['title']}'?"
            answer = entry['description']
            training_data.append({"question": question, "answer": answer})
    elif key == "informatik6klassebayern":
        for entry in new_data["informatik6klassebayern"]:
            question = f"Was ist das Thema '{entry['topic']}'?"
            answer = entry['description']
            training_data.append({"question": question, "answer": answer})
            for example in entry.get('examples', []):
                question = f"Was ist '{example['title']}'?"
                answer = example['summary']
                training_data.append({"question": question, "answer": answer})
    elif key == "mathematik6klassebayern":
        for entry in new_data["mathematik6klassebayern"]:
            question = f"Was ist das Thema '{entry['topic']}'?"
            answer = entry['description']
            training_data.append({"question": question, "answer": answer})
            for example in entry.get('examples', []):
                question = f"Was ist '{example['title']}'?"
                answer = example['summary']
                training_data.append({"question": question, "answer": answer})
    return training_data




def load_and_append_data(training_data, file_path, key):
    new_data = load_json_data(file_path)
    return append_data(training_data, new_data, key)
