import os
import sys
import winreg
import subprocess
import requests
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QHBoxLayout, QTextEdit, QMessageBox, QCheckBox, QSystemTrayIcon, QMenu, QAction
from PyQt5.QtGui import QPixmap, QFont, QPalette, QColor, QIcon
from PyQt5.QtCore import QProcess, Qt

def get_install_dir():
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\MINT-AI") as key:
            install_dir, _ = winreg.QueryValueEx(key, "InstallDir")
            return install_dir
    except FileNotFoundError:
        print("Installationsverzeichnis nicht gefunden")
        return None

def change_working_directory(install_dir):
    if install_dir:
        os.chdir(install_dir)
        print(f"Arbeitsverzeichnis erfolgreich zu {install_dir} gewechselt")
    else:
        print("Fehler beim Wechseln des Arbeitsverzeichnisses")

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def check_internet_connection():
    url = "http://www.google.com"
    timeout = 5
    try:
        request = requests.get(url, timeout=timeout)
        return True
    except (requests.ConnectionError, requests.Timeout) as exception:
        return False

install("PyQt5")

class LauncherGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('MINT AI Launcher')
        self.setGeometry(100, 100, 800, 600)
        
        # Hintergrundfarbe
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(50, 50, 50))
        self.setPalette(palette)
        
        # Layout
        layout = QVBoxLayout()

        # Ein-/Ausschalter für Schnellzugriffslogos
        self.logo_checkbox = QCheckBox("Schnellzugriffslogos anzeigen", self)
        self.logo_checkbox.setChecked(True)
        self.logo_checkbox.stateChanged.connect(self.toggle_logo)
        layout.addWidget(self.logo_checkbox)

        # Logo (falls vorhanden)
        self.logo_label = QLabel(self)
        self.logo_pixmap = QPixmap("./logo.png")
        self.logo_pixmap = self.logo_pixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.logo_label.setPixmap(self.logo_pixmap)
        self.logo_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.logo_label)
        
        # Titel
        self.title_label = QLabel('Welcome to MINT AI Launcher!', self)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setFont(QFont('Arial', 24))
        self.title_label.setStyleSheet("color: white;")
        layout.addWidget(self.title_label)
        
        # Verbindungssymbol
        self.connection_icon = QLabel(self)
        self.update_connection_icon()
        layout.addWidget(self.connection_icon)

        # Button zum Starten des Hauptprogramms
        self.start_button = QPushButton('Start MINT AI', self)
        self.start_button.setFont(QFont('Arial', 18))
        self.start_button.setStyleSheet("background-color: green; color: white; padding: 10px;")
        self.start_button.clicked.connect(self.start_program)
        layout.addWidget(self.start_button)
        
        # Button zum Suchen von Updates
        self.update_button = QPushButton('Updates suchen', self)
        self.update_button.setFont(QFont('Arial', 18))
        self.update_button.setStyleSheet("background-color: blue; color: white; padding: 10px;")
        self.update_button.clicked.connect(self.check_updates)
        layout.addWidget(self.update_button)
        
        # Button zum Beenden
        self.exit_button = QPushButton('Beenden', self)
        self.exit_button.setFont(QFont('Arial', 18))
        self.exit_button.setStyleSheet("background-color: red; color: white; padding: 10px;")
        self.exit_button.clicked.connect(self.close)
        layout.addWidget(self.exit_button)
        
        # Textbereich für Ausgabe
        self.text_area = QTextEdit(self)
        self.text_area.setReadOnly(True)
        self.text_area.setStyleSheet("background-color: black; color: white;")
        layout.addWidget(self.text_area)
        
        # Setze Layout
        layout.setAlignment(Qt.AlignCenter)
        self.setLayout(layout)

        # System Tray Icon erstellen
        self.tray_icon = QSystemTrayIcon(self)
        self.tray_icon.setIcon(QIcon("./logo.png"))
        
        # System Tray Icon Menü erstellen
        tray_menu = QMenu(self)
        show_action = QAction("Show", self)
        quit_action = QAction("Quit", self)
        show_action.triggered.connect(self.show)
        quit_action.triggered.connect(QApplication.instance().quit)
        tray_menu.addAction(show_action)
        tray_menu.addAction(quit_action)
        self.tray_icon.setContextMenu(tray_menu)
        
        # System Tray Icon anzeigen
        self.tray_icon.show()

        # QProcess zum Ausführen des Skripts
        self.process = QProcess(self)
        self.process.readyReadStandardOutput.connect(self.read_output)
        self.process.readyReadStandardError.connect(self.read_error)
        
    def toggle_logo(self, state):
        if state == Qt.Checked:
            self.logo_label.show()
        else:
            self.logo_label.hide()
    
    def update_connection_icon(self):
        if check_internet_connection():
            self.connection_icon.setPixmap(QPixmap("./wifi-strong.png"))
            self.connection_icon.setToolTip("Starke Internetverbindung")
        else:
            self.connection_icon.setPixmap(QPixmap("./wifi-weak.png"))
            self.connection_icon.setToolTip("Schwache oder keine Internetverbindung")
        
    def start_program(self):
        # Hauptprogramm starten
        self.text_area.append("Das Hauptprogramm wird gestartet...")  # Ausgabe im Textbereich
        try:
            self.process.start('python', ['test.py'])
        except Exception as e:
            QMessageBox.critical(self, "Fehler", f"Fehler beim Starten des Skripts: {e}")
    
    def check_updates(self):
        # Updates suchen
        if check_internet_connection():
            self.text_area.append("Nach Updates suchen...")  # Ausgabe im Textbereich
            try:
                result = subprocess.run(['python', 'update-isolated.py'], capture_output=True, text=True)
                self.text_area.append(result.stdout)
                if result.returncode != 0:
                    self.text_area.append(result.stderr)
            except Exception as e:
                QMessageBox.critical(self, "Fehler", f"Fehler beim Suchen nach Updates: {e}")
        else:
            self.text_area.append("Keine Internetverbindung. Updates konnten nicht gesucht werden.")
            QMessageBox.warning(self, "Keine Internetverbindung", "Es besteht keine Internetverbindung. Bitte stellen Sie eine Verbindung her, um nach Updates zu suchen.")
        
    def read_output(self):
        output = self.process.readAllStandardOutput().data().decode('latin-1')
        self.text_area.append(output)
        
    def read_error(self):
        error = self.process.readAllStandardError().data().decode('latin-1')
        self.text_area.append(error)

if __name__ == "__main__":
    install_dir = get_install_dir()
    change_working_directory(install_dir)

    app = QApplication(sys.argv)
    launcher = LauncherGUI()
    launcher.show()
    sys.exit(app.exec_())
