import os
import sys
import winreg
import subprocess
import requests
import tempfile
import shutil
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QHBoxLayout, QTextEdit, QMessageBox, QCheckBox, QSystemTrayIcon, QMenu, QAction, QComboBox, QFormLayout, QGroupBox, QInputDialog, QProgressBar, QFileDialog
from PyQt5.QtGui import QPixmap, QFont, QPalette, QColor, QIcon, QMovie
from PyQt5.QtCore import QProcess, Qt, QTimer

def load_addon(addon_path):
    if os.path.exists(addon_path):
        with open(addon_path, 'r') as file:
            exec(file.read(), globals())
    else:
        print(f"Addon {addon_path} nicht gefunden")

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

def read_registry_setting(key, value, default=None):
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key) as reg_key:
            result, _ = winreg.QueryValueEx(reg_key, value)
            return result
    except FileNotFoundError:
        return default

def write_registry_setting(key, value, data):
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, key) as reg_key:
        winreg.SetValueEx(reg_key, value, 0, winreg.REG_SZ, data)

install("PyQt5")

class LoadingScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('Laden...')
        self.setGeometry(100, 100, 400, 300)
        
        layout = QVBoxLayout()
        
        # Ladebalken
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)
        
        # Animiertes Logo
        self.animated_logo = QLabel(self)
        self.movie = QMovie("./logo.gif")
        self.animated_logo.setMovie(self.movie)
        layout.addWidget(self.animated_logo)
        
        self.setLayout(layout)
        
        self.movie.start()
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_progress)
        self.timer.start(50)  # Update alle 50ms

    def update_progress(self):
        value = self.progress_bar.value() + 1
        self.progress_bar.setValue(value)
        if value >= 100:
            self.timer.stop()
            self.close()
            self.start_main_app()

    def start_main_app(self):
        self.main_app = LauncherGUI()
        self.main_app.show()

class LauncherGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.load_dev_options()
        self.temp_addon_files = []

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
        
        # Button zum Installieren eines Addons
        self.install_addon_button = QPushButton('Addon installieren', self)
        self.install_addon_button.setFont(QFont('Arial', 18))
        self.install_addon_button.setStyleSheet("background-color: purple; color: white; padding: 10px;")
        self.install_addon_button.clicked.connect(self.install_addon)
        layout.addWidget(self.install_addon_button)
        
        # Entwickleroptionen
        self.dev_options_group = QGroupBox("Entwickleroptionen")
        self.dev_options_layout = QFormLayout()

        self.prevent_updates_checkbox = QCheckBox("Updates verhindern", self)
        self.prevent_updates_checkbox.stateChanged.connect(self.save_dev_options)
        self.dev_options_layout.addRow(self.prevent_updates_checkbox)

        self.logo_checkbox = QCheckBox("Schnellzugriffslogos deaktivieren", self)
        self.logo_checkbox.stateChanged.connect(self.save_dev_options)
        self.dev_options_layout.addRow(self.logo_checkbox)

        self.uninstall_button = QPushButton("Uninstall MINT-AI Launcher", self)
        self.uninstall_button.clicked.connect(self.uninstall_program)
        self.dev_options_layout.addRow(self.uninstall_button)

        self.GameButton = QPushButton("Start Game", self)
        self.GameButton.clicked.connect(self.start_game)
        self.dev_options_layout.addRow(self.GameButton)

        # Checkbox für Rendering-Option
        self.opengl_checkbox = QCheckBox("OpenGL3 Rendering verwenden", self)
        self.dev_options_layout.addRow(self.opengl_checkbox)

        self.dev_options_group.setLayout(self.dev_options_layout)
        layout.addWidget(self.dev_options_group)

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

    def install_addon(self):
        downloaded, ok = QInputDialog.getText(self, "Addon installieren", "Haben Sie ein Addon heruntergeladen? (ja/nein)")
        if ok and downloaded.lower() == 'ja':
            options = QFileDialog.Options()
            options |= QFileDialog.ReadOnly
            file_name, _ = QFileDialog.getOpenFileName(self, "Addon auswählen", "", "Addon Dateien (*.mintaiaddon);;Alle Dateien (*)", options=options)
            if file_name:
                load_addon(file_name)
                self.text_area.append(f"Addon {file_name} erfolgreich geladen.")
        else:
            # Auswahl zwischen den vorgegebenen Addons
            addon_choice, ok = QInputDialog.getItem(self, "Addon auswählen", "Wählen Sie ein Addon aus:", ["Blackbig", "Blueforever"], 0, False)
            if ok and addon_choice:
                if addon_choice == "Blackbig":
                    url = "https://raw.githubusercontent.com/Gymnasium-Freiham/MINT-AI-Addons/refs/heads/main/style-blackbig/addon-newstyle.mintaiaddon"
                elif addon_choice == "Blueforever":
                    url = "https://raw.githubusercontent.com/Gymnasium-Freiham/MINT-AI-Addons/refs/heads/main/style-blueforever/addon-newstyle.mintaiaddon"
                self.download_and_load_addon(url)

    def download_and_load_addon(self, url):
        try:
            response = requests.get(url)
            response.raise_for_status()
            addon_code = response.text
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mintaiaddon")
            with open(temp_file.name, 'w') as file:
                file.write(addon_code)
            self.temp_addon_files.append(temp_file.name)
            load_addon(temp_file.name)
            self.text_area.append(f"Addon von {url} erfolgreich geladen.")
        except requests.RequestException as e:
            QMessageBox.critical(self, "Fehler", f"Fehler beim Herunterladen des Addons: {e}")

    def closeEvent(self, event):
        # Temporäre Addon-Dateien löschen
        for temp_file in self.temp_addon_files:
            try:
                os.remove(temp_file)
            except OSError as e:
                print(f"Fehler beim Löschen der temporären Datei {temp_file}: {e}")
        event.accept()


    def start_game(self):
        # Passwortabfrage
        password, ok = QInputDialog.getText(self, 'Passwort eingeben', 'Bitte geben Sie das Passwort ein:')
        
        if ok and password == '9999':
            self.text_area.append("Das Game wird gestartet...")  # Ausgabe im Textbereich
            try:
                if self.opengl_checkbox.isChecked():
                    self.process.start("./Bowling-jump-new.exe", ["--rendering-driver", "opengl3"])
                else:
                    self.process.start("./Bowling-jump-new.exe")
            except Exception as e:
                QMessageBox.critical(self, "Fehler", f"Fehler beim Starten des Skripts: {e}")
        else:
            QMessageBox.warning(self, "Falsches Passwort", "Das eingegebene Passwort ist falsch.")

    def uninstall_program(self):
        reply = QMessageBox.question(self, 'Bestätigung', 
                                     'Sind Sie sicher, dass der MINT-AI-Launcher deinstalliert wird?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            reply = QMessageBox.question(self, 'Bestätigung', 
                                         'Sind Sie wirklich sicher, dass der MINT-AI-Launcher deinstalliert wird?',
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
            if reply == QMessageBox.Yes:
                reply = QMessageBox.question(self, 'Bestätigung', 
                                             'Ist es sicher Ihre letzte Entscheidung?',
                                             QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            
                if reply == QMessageBox.Yes:
                    try:
                        os.system("sudo ./Uninstall.exe")
                        QMessageBox.information(self, "Deinstallation", "MINT-AI-Launcher wurde erfolgreich deinstalliert.")
                        self.close()
                    except Exception as e:
                        QMessageBox.critical(self, "Fehler", f"Fehler bei der Deinstallation: {e}")  
    def update_connection_icon(self):
        if check_internet_connection():
            self.connection_icon.setPixmap(QPixmap("./wifi-strong.png"))
            self.connection_icon.setToolTip("Starke Internetverbindung")
        else:
            self.connection_icon.setPixmap("./wifi-weak.png")
            self.connection_icon.setToolTip("Schwache oder keine Internetverbindung")
    
    def start_program(self):
        # Hauptprogramm starten
        self.text_area.append("Das Hauptprogramm wird gestartet...")  # Ausgabe im Textbereich
        try:
            self.process.start('python', ['test.py'])
        except Exception as e:
            QMessageBox.critical(self, "Fehler", f"Fehler beim Starten des Skripts: {e}")
    
    def check_updates(self):
        # Updates suchensa
        if self.prevent_updates_checkbox.isChecked():
            self.text_area.append("Updates sind derzeit deaktiviert.")
            QMessageBox.information(self, "Updates deaktiviert", "Updates sind in den Entwickleroptionen deaktiviert.")
            return

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
    
    def save_dev_options(self):
        write_registry_setting(r"Software\MINT-AI", "PreventUpdates", "True" if self.prevent_updates_checkbox.isChecked() else "False")
        write_registry_setting(r"Software\MINT-AI", "DisableLogos", "True" if self.logo_checkbox.isChecked() else "False")
        self.toggle_logo()

    def toggle_logo(self):
        if self.logo_checkbox.isChecked():
            self.logo_label.hide()
        else:
            self.logo_label.show()
    
    def load_dev_options(self):
        if read_registry_setting(r"Software\MINT-AI", "PreventUpdates", "False") == "True":
            self.prevent_updates_checkbox.setChecked(True)
        if read_registry_setting(r"Software\MINT-AI", "DisableLogos", "False") == "True":
            self.logo_checkbox.setChecked(True)
            self.logo_label.hide()

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
    
    # Ladebildschirm anzeigen
    loading_screen = LoadingScreen()
    loading_screen.show()

    # Addon ausführen
    addon_path = os.path.join(install_dir, 'addons', 'addon-newstyle.mintaiaddon')
    if os.path.exists(addon_path):
        load_addon(addon_path)
    else:
        print(f"Addon {addon_path} nicht gefunden")
    
    sys.exit(app.exec_())
