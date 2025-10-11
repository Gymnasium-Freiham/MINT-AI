# -*- coding: utf-8 -*-
import os
import requests

# GitHub API URL für das Repository
repo_url = "https://api.github.com/repos/Gymnasium-Freiham/LATIN-AI/contents/"

# Directory where files will be saved
directory = "./"

# Ensure the directory exists
if not os.path.exists(directory):
    os.makedirs(directory)

def download_file(url, filename):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        file_path = os.path.join(directory, filename)
        with open(file_path, 'wb') as file:
            total_length = int(response.headers.get('content-length', 0))
            downloaded = 0
            for chunk in response.iter_content(1024):
                if chunk:
                    file.write(chunk)
                    downloaded += len(chunk)
                    progress = int(100 * downloaded / total_length)
                    print(f'Progress: {progress}')
        print(f'Datei {filename} erfolgreich heruntergeladen und gespeichert.')
    else:
        print(f'Fehler beim Herunterladen der Datei: Status-Code {response.status_code}')

def fetch_and_download_files(api_url, path=""):
    response = requests.get(api_url)
    if response.status_code == 200:
        contents = response.json()
        for item in contents:
            if item['type'] == 'file':
                file_url = item['download_url']
                file_path = os.path.join(path, item['name'])
                download_file(file_url, file_path)
            elif item['type'] == 'dir':
                new_path = os.path.join(path, item['name'])
                if not os.path.exists(os.path.join(directory, new_path)):
                    os.makedirs(os.path.join(directory, new_path))
                fetch_and_download_files(item['url'], new_path)
    else:
        print(f'Fehler beim Abrufen des Repository-Inhalts: Status-Code {response.status_code}')

# Fetch and download all files from the repository
fetch_and_download_files(repo_url)
