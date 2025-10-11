import os
import sys
import winreg
import subprocess
import tempfile
import shutil
import webbrowser
from urllib.parse import urlparse, parse_qs
import socket

# Prüfen, ob der Parameter --no-connection übergeben wurde
no_connection = "--no-connection" in sys.argv
GIBBERLINK_EXPERIMENTAL = "--gibberlink=true" in sys.argv  # oder False, je nach Bedarf
# Setze eine Umgebungsvariable, die von anderen Skripten gelesen werden kann
if no_connection:
    os.environ["NO_CONNECTION"] = "1"
else:
    os.environ["NO_CONNECTION"] = "0"
def install(package):
    if no_connection:
        # no network allowed
        return
    try:
        # suppress pip output to avoid spamming the console
        subprocess.run([sys.executable, "-m", "pip", "install", package],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except subprocess.CalledProcessError:
        # ignore install errors here; caller can handle logging if needed
        pass


def check_internet_connection(timeout=3):
    try:
        # Versuche, eine Verbindung zu einem bekannten Server (Google DNS)
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("8.8.8.8", 53))
        return True
    except socket.error:
        return False
try:
    import PyQt5
except ImportError:
    install("PyQt5")
try:
    import requests
except ImportError:
    install("requests")
try:
    import Flask
except ImportError:
    install("Flask")
try:
    import Flask_CORS
except ImportError:
    install("Flask-CORS")
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QHBoxLayout, QTextEdit, QMessageBox, QCheckBox, QSystemTrayIcon, QMenu, QAction, QComboBox, QFormLayout, QGroupBox, QInputDialog, QProgressBar, QFileDialog
from PyQt5.QtGui import QPixmap, QFont, QPalette, QColor, QIcon, QMovie
from PyQt5.QtCore import QProcess, Qt, QTimer, QEvent, QThread, pyqtSignal
import requests

class DependencyInstaller(QThread):
    progress_updated = pyqtSignal(int)  # Signal für Fortschrittsaktualisierung
    installation_finished = pyqtSignal()  # Signal, wenn die Installation abgeschlossen ist

    def run(self):
        dependencies = ["PyQt5", "requests", "Flask", "Flask-CORS"]
        total = len(dependencies)
        for i, package in enumerate(dependencies, start=1):
            if no_connection:
                # skip network installs
                pass
            else:
                try:
                    subprocess.run([sys.executable, "-m", "pip", "install", package],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                except Exception as e:
                    # don't spam UI; keep silent on install failures here
                    pass
            progress = int((i / total) * 100)
            self.progress_updated.emit(progress)  # Fortschritt aktualisieren
        self.installation_finished.emit()  # Installation abgeschlossen

def register_url_scheme():
    try:
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, r"Software\Classes\mintai") as key:
            winreg.SetValueEx(key, "", 0, winreg.REG_SZ, "URL:mintai Protocol")
            winreg.SetValueEx(key, "URL Protocol", 0, winreg.REG_SZ, "")
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, r"Software\Classes\mintai\shell\open\command") as key:
            winreg.SetValueEx(key, "", 0, winreg.REG_SZ, f'"{sys.executable}" "{os.path.abspath(__file__)}" "%1"')
        print("URL-Schema 'mintai' erfolgreich registriert")
    except Exception as e:
        print(f"Fehler beim Registrieren des URL-Schemas: {e}")


def handle_custom_url(url):
    parsed_url = urlparse(url)
    if parsed_url.scheme == 'mintai' and parsed_url.netloc == 'install-addon':
        query_params = parse_qs(parsed_url.query)
        addon_url = query_params.get('url', [None])[0]
        if addon_url:
            app.main_window.download_and_load_addon(addon_url)

def open_addon_store():
    store_path = os.path.join(os.path.dirname(__file__), './browser-powered/addon_store.html')
    webbrowser.open(f'file://{store_path}')

def load_addon(addon_path):
    if os.path.exists(addon_path):
        with open(addon_path, 'r') as file:
            exec(file.read(), globals())
    else:
        print(f"Addon {addon_path} nicht gefunden")

def load_all_addons(addons_dir):
    if os.path.exists(addons_dir) and os.path.isdir(addons_dir):
        for file_name in os.listdir(addons_dir):
            if file_name.endswith('.mintaiaddon'):
                addon_path = os.path.join(addons_dir, file_name)
                load_addon(addon_path)
    else:
        print(f"Addons-Verzeichnis {addons_dir} nicht gefunden oder ist kein Verzeichnis")

def get_install_dir():
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\LATIN AI") as key:
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

class LoadingScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.movie = None  # Initialisiere self.movie als Instanzattribut
        self.initUI()
        self.installer_thread = DependencyInstaller()
        self.installer_thread.progress_updated.connect(self.update_progress)
        self.installer_thread.installation_finished.connect(self.on_installation_finished)

    def initUI(self):
        self.setWindowTitle('Laden...')
        self.showFullScreen()  # Ladebildschirm im Vollbildmodus anzeigen
        
        layout = QVBoxLayout()
        
        # Ladebalken
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)
        
        # Animiertes Logo
        self.animated_logo = QLabel(self)
        self.movie = QMovie("./logo.gif")  # Initialisiere das QMovie-Objekt
        self.animated_logo.setMovie(self.movie)
        self.animated_logo.setAlignment(Qt.AlignCenter)  # Zentriere das GIF
        layout.addWidget(self.animated_logo)
        
        self.setLayout(layout)
        
        self.movie.start()

    def resizeEvent(self, event):
        """Skaliere das GIF und das QLabel, wenn das Fenster die Größe ändert."""
        if self.movie:  # Überprüfe, ob self.movie korrekt initialisiert wurde
            size = self.size()  # Verwende die Fenstergröße
            self.animated_logo.resize(size)  # Passe die Größe des QLabel an
            self.movie.setScaledSize(size)  # Passe die Größe des GIFs an das QLabel an
        super().resizeEvent(event)

    def start_installation(self):
        self.installer_thread.start()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def on_installation_finished(self):
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
        # track a short-lived launch state to avoid rapid repeated clicks
        self._launch_in_progress = False
        app.installEventFilter(self)

    def initUI(self):
        self.setWindowTitle('LATIN AI Launcher')
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
        self.title_label = QLabel('Welcome to LATIN AI Launcher!', self)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setFont(QFont('Arial', 24))
        self.title_label.setStyleSheet("color: white;")
        layout.addWidget(self.title_label)
        

        # Button zum Starten des Hauptprogramms
        self.start_button = QPushButton('Start LATIN AI', self)
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
        
        # Button zum Öffnen des Addon-Stores
        self.open_store_button = QPushButton('Addon Store öffnen', self)
        self.open_store_button.setFont(QFont('Arial', 18))
        self.open_store_button.setStyleSheet("background-color: orange; color: white; padding: 10px;")
        self.open_store_button.clicked.connect(open_addon_store)
        layout.addWidget(self.open_store_button)
        
        # Entwickleroptionen
        self.dev_options_group = QGroupBox("Entwickleroptionen")
        self.dev_options_layout = QFormLayout()

        self.prevent_updates_checkbox = QCheckBox("Updates verhindern", self)
        self.prevent_updates_checkbox.stateChanged.connect(self.save_dev_options)
        self.dev_options_layout.addRow(self.prevent_updates_checkbox)

        self.logo_checkbox = QCheckBox("Schnellzugriffslogos deaktivieren", self)
        self.logo_checkbox.stateChanged.connect(self.save_dev_options)
        self.dev_options_layout.addRow(self.logo_checkbox)

        self.gibberlink_checkbox = QCheckBox("Gibberlink aktivieren", self)
        self.gibberlink_checkbox.stateChanged.connect(self.save_dev_options)
        self.dev_options_layout.addRow(self.gibberlink_checkbox)

        self.uninstall_button = QPushButton("Uninstall LATIN-AI Launcher", self)
        self.uninstall_button.clicked.connect(self.uninstall_program)
        self.dev_options_layout.addRow(self.uninstall_button)

        # Button zum Deinstallieren eines Addons
        self.uninstall_addon_button = QPushButton('Addon deinstallieren', self)
        self.uninstall_addon_button.setFont(QFont('Arial', 18))
        self.uninstall_addon_button.setStyleSheet("background-color: red; color: white; padding: 5px;")
        self.uninstall_addon_button.clicked.connect(self.uninstall_addon)
        self.dev_options_layout.addRow(self.uninstall_addon_button)

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

    def uninstall_addon(self):
        install_dir = get_install_dir()
        addons_dir = os.path.join(install_dir, 'addons')
        if not os.path.exists(addons_dir):
            QMessageBox.warning(self, "Fehler", "Keine Addons installiert.")
            return

        addons = [f for f in os.listdir(addons_dir) if f.endswith('.mintaiaddon')]
        if not addons:
            QMessageBox.warning(self, "Fehler", "Keine Addons installiert.")
            return

        addon_choice, ok = QInputDialog.getItem(self, "Addon deinstallieren", "Wählen Sie ein Addon aus:", addons, 0, False)
        if ok and addon_choice:
            addon_path = os.path.join(addons_dir, addon_choice)
            try:
                os.remove(addon_path)
                self.text_area.append(f"Addon {addon_choice} erfolgreich deinstalliert.")
            except Exception as e:
                QMessageBox.critical(self, "Fehler", f"Fehler beim Deinstallieren des Addons: {e}")


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
            addon_choice, ok = QInputDialog.getItem(self, "Addon auswählen", "Wählen Sie ein Addon aus:", ["Blackbig", "Blueforever", "Dark", "Light", "Blue", "Green", "Red"], 0, False)
            if ok and addon_choice:
                if addon_choice == "Blackbig":
                    url = "https://raw.githubusercontent.com/Gymnasium-Freiham/MINT-AI-Addons/refs/heads/main/style-blackbig/addon-newstyle.mintaiaddon"
                elif addon_choice == "Blueforever":
                    url = "https://raw.githubusercontent.com/Gymnasium-Freiham/MINT-AI-Addons/refs/heads/main/style-blueforever/addon-newstyle.mintaiaddon"
                elif addon_choice == "Dark":
                    url = "https://raw.githubusercontent.com/Gymnasium-Freiham/MINT-AI-Addons/refs/heads/main/style-dark/addon-newstyle.mintaiaddon"
                elif addon_choice == "Light":
                    url = "https://raw.githubusercontent.com/Gymnasium-Freiham/MINT-AI-Addons/refs/heads/main/style-light/addon-newstyle.mintaiaddon"
                elif addon_choice == "Blue":
                    url = "https://raw.githubusercontent.com/Gymnasium-Freiham/MINT-AI-Addons/refs/heads/main/style-blue/addon-newstyle.mintaiaddon"
                elif addon_choice == "Green":
                    url = "https://raw.githubusercontent.com/Gymnasium-Freiham/MINT-AI-Addons/refs/heads/main/style-green/addon-newstyle.mintaiaddon"
                elif addon_choice == "Red":
                    url = "https://raw.githubusercontent.com/Gymnasium-Freiham/MINT-AI-Addons/refs/heads/main/style-red/addon-newstyle.mintaiaddon"
                self.download_and_load_addon(url)

    def download_and_load_addon(self, url):
        try:
            response = requests.get(url)
            response.raise_for_status()
            addon_code = response.text
            install_dir = get_install_dir()
            addons_dir = os.path.join(install_dir, 'addons')
            if not os.path.exists(addons_dir):
                os.makedirs(addons_dir)
            addon_name = os.path.basename(url)
            addon_path = os.path.join(addons_dir, addon_name)
            with open(addon_path, 'w') as file:
                file.write(addon_code)
            self.text_area.append(f"Addon von {url} erfolgreich heruntergeladen und gespeichert.")
            self.start_program()
        except requests.RequestException as e:
            QMessageBox.critical(self, "Fehler", f"Fehler beim Herunterladen des Addons: {e}")



    def apply_addon_effects(self, addon_file):
        # Beispiel: Ändern Sie die Hintergrundfarbe basierend auf dem geladenen Addon
        if "blueforever" in addon_file.lower():
            palette = self.palette()
            palette.setColor(QPalette.Window, QColor(0, 0, 255))  # Blau
            self.setPalette(palette)
        elif "blackbig" in addon_file.lower():
            palette = self.palette()
            palette.setColor(QPalette.Window, QColor(0, 0, 0))  # Schwarz
            self.setPalette(palette)
        # Fügen Sie hier weitere Addon-Effekte hinzu

    def closeEvent(self, event):
        # Temporäre Addon-Dateien löschen
        for temp_file in self.temp_addon_files:
            try:
                os.remove(temp_file)
            except OSError as e:
                print(f"Fehler beim Löschen der temporären Datei {temp_file}: {e}")
        event.accept()

    def set_main_window(window):
        app.main_window = window
        app.main_window.show()
    
    def eventFilter(self, obj, event):
        if event.type() == QEvent.FileOpen:
            url = event.url().toString()
            handle_custom_url(url)
            return True
        return super().eventFilter(obj, event)

    def uninstall_program(self):
        reply = QMessageBox.question(self, 'Bestätigung', 
        'Sind Sie sicher, dass der LATIN-AI-Launcher deinstalliert werden soll?',
        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            reply = QMessageBox.question(self, 'Bestätigung', 
                                        'Sind Sie wirklich sicher, dass der LATIN-AI-Launcher deinstalliert werden soll?',
                                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
            if reply == QMessageBox.Yes:
                reply = QMessageBox.question(self, 'Bestätigung', 
                                             'Ist es sicher Ihre letzte Entscheidung?',
                                             QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            
                if reply == QMessageBox.Yes:
                    try:
                        os.system("sudo ./Uninstall.exe")
                        QMessageBox.information(self, "Deinstallation", "LATIN-AI-Launcher wurde erfolgreich deinstalliert.")
                        self.close()
                    except Exception as e:
                        QMessageBox.critical(self, "Fehler", f"Fehler bei der Deinstallation: {e}")  
 
    def start_program(self):
        # Hauptprogramm starten
        self.text_area.append("Das Hauptprogramm wird gestartet...")  # Ausgabe im Textbereich
        # prevent very rapid repeated launches
        if self._launch_in_progress:
            self.text_area.append("Start wird bereits ausgeführt. Bitte warten...")
            return
        self._launch_in_progress = True
        self.start_button.setEnabled(False)
        program = sys.executable
        args = ['test.py'] + (['--gibberlink=true'] if GIBBERLINK_EXPERIMENTAL else [])
        try:
            # Start detached so launcher does NOT capture child's stdout/stderr
            QProcess.startDetached(program, args)
            self.text_area.append("Hauptprogramm (test.py) im Hintergrund gestartet.")
        except Exception as e:
            QMessageBox.critical(self, "Fehler", f"Fehler beim Starten des Skripts: {e}")
        # re-enable button after a short interval to prevent accidental flooding
        QTimer.singleShot(1000, lambda: (setattr(self, "_launch_in_progress", False), self.start_button.setEnabled(True)))

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
        GIBBERLINK_EXPERIMENTAL= True if self.gibberlink_checkbox.isChecked() else False
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
        proc = self.sender()
        if proc is None:
            return
        try:
            out = proc.readAllStandardOutput().data().decode('latin-1')
            if out:
                self.text_area.append(out)
        except Exception:
            pass
    def read_error(self):
        proc = self.sender()
        if proc is None:
            return
        try:
            err = proc.readAllStandardError().data().decode('latin-1')
            if err:
                self.text_area.append(err)
        except Exception:
            pass

    
def check_and_start_server():
    try:
        response = requests.get('http://localhost:5001/install_dir')
        if response.status_code == 200:
            print("Server läuft bereits auf Port 5001")
            return
    except requests.ConnectionError:
        print("Server läuft nicht, starte server.py")
        # avoid invalid escape sequences and build a safe path
        os.startfile(os.path.join(install_dir, 'server.py'))

if __name__ == "__main__":
    install_dir = get_install_dir()
    check_and_start_server()
    change_working_directory(install_dir)
    register_url_scheme()

    app = QApplication(sys.argv)
    # Alle Addons ausführen
    addons_dir = os.path.join(install_dir, 'addons')
    load_all_addons(addons_dir)
    # Ladebildschirm anzeigen und Installation starten
    loading_screen = LoadingScreen()
    loading_screen.show()
    loading_screen.start_installation()


    sys.exit(app.exec_())

        # Alle Addons ausführen
    addons_dir = os.path.join(install_dir, 'addons')
    load_all_addons(addons_dir)
