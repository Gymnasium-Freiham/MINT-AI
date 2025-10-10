# -*- coding: utf-8 -*-
import subprocess
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLineEdit, QPushButton, QTextEdit, QHBoxLayout, QLabel, QComboBox, QStatusBar, QMainWindow, QProgressBar
from PyQt5.QtGui import QFont, QPalette, QColor, QPixmap, QIcon
from PyQt5.QtCore import QProcess, Qt, QTimer
GIBBERLINK_EXPERIMENTAL = "--gibberlink=true" in sys.argv  # oder False, je nach Bedarf
class TerminalGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('LATIN AI')
        self.setGeometry(100, 100, 800, 600)
        
        # Anwendungssymbol festlegen
        self.setWindowIcon(QIcon('./icon.png'))
        
        # Hintergrundfarbe einstellen
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(45, 45, 45))
        self.setPalette(palette)
        
        # Zentrales Widget und Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Ueberschrift mit Logo
        self.logo_label = QLabel(self)
        self.logo_pixmap = QPixmap("./logo.png")
        self.logo_pixmap = self.logo_pixmap.scaled(300, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.logo_label.setPixmap(self.logo_pixmap)
        self.logo_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.logo_label)
        
        # Textbereich fuer Ausgabe
        self.text_area = QTextEdit(self)
        self.text_area.setReadOnly(True)
        self.text_area.setStyleSheet("background-color: #003366; color: white; font-family: Consolas; font-size: 16px;")
        layout.addWidget(self.text_area)
        
        # Eingabefeld und Sendebutton
        input_layout = QHBoxLayout()
        self.entry = QLineEdit(self)
        self.entry.setPlaceholderText("Geben Sie hier Ihre Eingabe ein...")
        self.entry.setToolTip("Hier koennen Sie Ihren Text eingeben und senden")
        self.entry.setStyleSheet("background-color: #004080; color: white; font-family: Consolas; font-size: 16px;")
        self.send_button = QPushButton('Senden', self)
        self.send_button.setToolTip("Klicken Sie hier, um Ihre Eingabe zu senden")
        self.send_button.setStyleSheet("background-color: #007acc; color: white; font-size: 16px; padding: 8px;")
        self.send_button.clicked.connect(self.send_input)
        input_layout.addWidget(self.entry)
        input_layout.addWidget(self.send_button)
        layout.addLayout(input_layout)
        
        # Dropdown-Menue fuer Themenwechsel
        self.theme_selector = QComboBox(self)
        self.theme_selector.addItems(["Dark Theme", "Light Theme", "Blue Theme"])
        self.theme_selector.currentIndexChanged.connect(self.change_theme)
        layout.addWidget(self.theme_selector)
        
        # Fortschrittsanzeige
        self.progress_bar = QProgressBar(self)
        layout.addWidget(self.progress_bar)
        
        # Statusleiste
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage('Bereit')
        self.status_bar.setStyleSheet("background-color: green; color: white;")
        
        self.setLayout(layout)
        
        # QProcess zum Ausfuehren des Skripts
        self.process = QProcess(self)
        self.process.readyReadStandardOutput.connect(self.read_output)
        self.process.readyReadStandardError.connect(self.read_error)
        self.process.started.connect(self.process_started)
        self.process.finished.connect(self.process_finished)
        try:
            subprocess.Popen(['python', 'main.py'] + (['--gibberlink=true'] if GIBBERLINK_EXPERIMENTAL else []))
        except Exception as e:
            self.text_area.append(f"Fehler beim Starten des Skripts: {e}")
            self.status_bar.setStyleSheet("background-color: red; color: white;")
            self.status_bar.showMessage('Fehler beim Starten des Hauptprogramms')
        
    def process_started(self):
        self.status_bar.setStyleSheet("background-color: blue; color: white;")
        self.status_bar.showMessage('Hauptprogramm gestartet')

    def process_finished(self):
        self.status_bar.setStyleSheet("background-color: green; color: white;")
        self.status_bar.showMessage('Hauptprogramm beendet')

    def read_output(self):
        output = self.process.readAllStandardOutput().data().decode('latin-1')
        self.text_area.append(output)
        self.update_progress(100)
        QTimer.singleShot(500, self.reset_progress)
        self.status_bar.setStyleSheet("background-color: yellow; color: black;")
        self.status_bar.showMessage('Ausgabe erhalten')
        
    def read_error(self):
        error = self.process.readAllStandardError().data().decode('latin-1')
        self.text_area.append(error)
        self.update_progress(100)
        QTimer.singleShot(500, self.reset_progress)
        self.status_bar.setStyleSheet("background-color: red; color: white;")
        self.status_bar.showMessage('Fehler erhalten')
        
    def send_input(self):
        user_input = self.entry.text()
        # Only write to the process if it's running
        if hasattr(self, 'process') and self.process.state() == QProcess.Running:
            self.process.write(user_input.encode() + b'\n')
        else:
            self.text_area.append("Kein Backend-Prozess verbunden; Eingabe nicht gesendet.")
        self.entry.clear()
        self.update_progress(100)
        QTimer.singleShot(500, self.reset_progress)
        self.status_bar.setStyleSheet("background-color: green; color: white;")
        self.status_bar.showMessage('Eingabe gesendet')
        
    def update_progress(self, value):
        current_value = self.progress_bar.value()
        while current_value < value:
            current_value += 1
            self.progress_bar.setValue(current_value)
            QTimer.singleShot(10, lambda: None)  # Wartezeit einfügen, um die Animation zu zeigen

    def reset_progress(self):
        self.progress_bar.setValue(0)

    def change_theme(self, index):
        if index == 0:  # Dark Theme
            self.setStyleSheet("")
            self.text_area.setStyleSheet("background-color: #1e1e1e; color: #d4d4d4; font-family: Consolas; font-size: 16px;")
            self.entry.setStyleSheet("background-color: #2e2e2e; color: #d4d4d4; font-family: Consolas; font-size: 16px;")
            self.send_button.setStyleSheet("background-color: #007acc; color: white; font-size: 16px; padding: 8px;")
            self.status_bar.showMessage('Dark Theme ausgewaehlt')
        elif index == 1:  # Light Theme
            self.setStyleSheet("background-color: #f0f0f0;")
            self.text_area.setStyleSheet("background-color: white; color: black; font-family: Consolas; font-size: 16px;")
            self.entry.setStyleSheet("background-color: #ffffff; color: black; font-family: Consolas; font-size: 16px;")
            self.send_button.setStyleSheet("background-color: #007acc; color: white; font-size: 16px; padding: 8px;")
            self.status_bar.showMessage('Light Theme ausgewaehlt')
        elif index == 2:  # Blue Theme
            self.setStyleSheet("background-color: #001f3f;")
            self.text_area.setStyleSheet("background-color: #003366; color: white; font-family: Consolas; font-size: 16px;")
            self.entry.setStyleSheet("background-color: #004080; color: white; font-family: Consolas; font-size: 16px;")
            self.send_button.setStyleSheet("background-color: #007acc; color: white; font-size: 16px; padding: 8px;")
            self.status_bar.showMessage('Blue Theme ausgewaehlt')

if __name__ == "__main__":
    app = QApplication(sys.argv)
    terminal_gui = TerminalGUI()
    terminal_gui.show()
    sys.exit(app.exec_())
