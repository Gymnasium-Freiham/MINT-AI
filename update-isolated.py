# main.py
import os
import sys
import subprocess
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QProgressBar, QStatusBar, QPushButton
from PyQt5.QtCore import Qt, QThread, pyqtSignal

class UpdateThread(QThread):
    update_progress = pyqtSignal(int)
    update_status = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.process = None

    def run(self):
        self.process = subprocess.Popen([sys.executable, "update.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for line in self.process.stdout:
            if "Progress" in line:
                progress = int(line.strip().split(':')[1].replace('%', '').strip())
                self.update_progress.emit(progress)
            else:
                self.update_status.emit(line.strip())
        self.process.wait()

class UpdateGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.thread = UpdateThread()
        self.thread.update_progress.connect(self.update_progress_bar)
        self.thread.update_status.connect(self.update_status_bar)

    def initUI(self):
        self.setWindowTitle('Update Status')
        self.setGeometry(200, 200, 400, 200)

        layout = QVBoxLayout()

        self.status_label = QLabel('Updates werden heruntergeladen...', self)
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)

        self.status_bar = QStatusBar(self)
        layout.addWidget(self.status_bar)

        self.update_button = QPushButton("Updates starten", self)
        self.update_button.clicked.connect(self.start_update)
        layout.addWidget(self.update_button)

        self.setLayout(layout)

    def start_update(self):
        self.thread.start()

    def update_progress_bar(self, progress):
        self.progress_bar.setValue(progress)

    def update_status_bar(self, status):
        self.status_bar.showMessage(status)
        self.status_label.setText(status)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    update_gui = UpdateGUI()
    update_gui.show()
    sys.exit(app.exec_())
