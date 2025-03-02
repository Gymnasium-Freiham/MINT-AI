# -*- coding: utf-8 -*-
print("Do not close this window! The addon store can't work without. Thank you. MINT AI Software.")
from flask import Flask, jsonify, request
from flask_cors import CORS
import winreg
import os
import subprocess
import logging

app = Flask(__name__)
CORS(app)  # CORS für alle Routen aktivieren

# Logging konfigurieren
logging.basicConfig(level=logging.DEBUG)

def get_install_dir():
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\MINT-AI") as key:
            install_dir, _ = winreg.QueryValueEx(key, "InstallDir")
            logging.debug(f"Installationsverzeichnis: {install_dir}")
            return install_dir
    except FileNotFoundError:
        logging.error("Installationsverzeichnis nicht gefunden")
        return None

@app.route('/install_dir', methods=['GET'])
def install_dir():
    install_dir = get_install_dir()
    if install_dir:
        return jsonify({"install_dir": install_dir})
    else:
        return jsonify({"error": "Installationsverzeichnis nicht gefunden"}), 404

@app.route('/save_addon', methods=['POST'])
def save_addon():
    data = request.json
    path = data.get('path')
    code = data.get('code')
    try:
        logging.debug(f"Speichern des Addons an Pfad: {path}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as file:
            file.write(code)
        return jsonify({"message": "Addon gespeichert"})
    except Exception as e:
        logging.error(f"Fehler beim Speichern des Addons: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/start_launcher', methods=['POST'])
def start_launcher():
    data = request.json
    install_dir = data.get('install_dir')
    try:
        logging.debug(f"Starten des Launchers aus Verzeichnis: {install_dir}")
        subprocess.Popen(['python', os.path.join(install_dir, 'launcher.py')])
        return jsonify({"message": "Launcher gestartet"})
    except Exception as e:
        logging.error(f"Fehler beim Starten des Launchers: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5001)
