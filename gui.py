import sys

from PySide6.QtCore import Qt, Slot, QBasicTimer
from PySide6.QtWidgets import QMainWindow, QApplication, QWidget, QPushButton, QLineEdit, QMessageBox, QLabel, QGridLayout, QSizePolicy
from PySide6.QtGui import QPixmap
import pyqtgraph as pg

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setGeometry(0,0,1000,800)
        self.setWindowTitle("Small Target Tracker")
        self.grid = QGridLayout()
        self.setLayout(self.grid)
        
        self.initUI()
        self.show()

    def initUI(self):
        # Parameters that can be changed
        self.message1 = QLabel("<h3>Parameters</h3>",self)
        self.message1.move(0,0)

        self.message2 = QLabel("Parameter 1",self)
        self.textbox = QLineEdit('20',self)
        self.message2.move(5,50)
        self.textbox.move(5, 75)
        self.textbox.resize(100,30)

        self.message3 = QLabel("Parameter 2",self)
        self.textbox2 = QLineEdit('20',self)
        self.message3.move(5,125)
        self.textbox2.move(5, 150)
        self.textbox2.resize(100,30)

        # Button to load image
        self.button1 = QPushButton('Load Image Frames',self)
        self.button1.move(5, 500)
        self.button1.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.button1.resize(150,70)

        # Button interactions
        self.button1.clicked.connect(self.load)

    @Slot()
    def load(self):
        self.label = QLabel(self)
        self.label.move(200, 50)
        self.label.resize(600,400)
        self.timer = QBasicTimer()
        self.step = 0
        self.delay = 1000
        # Need to add data loader here, should return a bunch of things to display here.
        self.img = ["img/000001.jpg","img/000030.jpg","img/000060.jpg"]

        if self.timer.isActive():
            self.timer.stop()
        else:
            self.timer.start(self.delay, self)

        self.graphData()
        return 

    def timerEvent(self, event):
        if self.step >= len(self.img):
            self.timer.stop()
            self.step = 0
            return

        file = self.img[self.step]
        image = QPixmap(file).scaled(self.label.size(), Qt.KeepAspectRatio)
        self.label.setPixmap(image)
        self.label.show()
        self.step += 1

    def graphData(self):
        pen = pg.mkPen(width = 10)
        self.graphWidget = pg.PlotWidget()
        self.graphWidget.show()
        seconds = [1,2,3,4,5,6,7,8,9,10]
        objects = [30,32,34,32,33,31,29,32,35,45]

        self.graphWidget.plot(seconds, objects, pen=pen)
        self.graphWidget.setTitle("Moving objects per frame", size = "20pt")
        self.graphWidget.setLabel('left', 'Average Moving Object (per frame)')
        self.graphWidget.setLabel('bottom', 'Seconds (s)')



if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = App()
    sys.exit(app.exec())