import sys
import time
from dataparser import Dataloader

from os import listdir
from os.path import isfile, join
from PySide6.QtCore import Qt, Slot, QBasicTimer, QStringListModel
from PySide6.QtWidgets import QMainWindow, QApplication, QWidget, QPushButton, QLineEdit, QMessageBox, QLabel, QGridLayout, QSizePolicy, QFileDialog
from PySide6.QtGui import QPixmap, QPainter, QPen
from PIL import Image, ImageQt
import pyqtgraph as pg

class Control(QWidget):
    def __init__(self, parent = None):
        super(Control, self).__init__(parent)
        self.setWindowTitle("Small Target Tracker")
        self.grid = QGridLayout()
        self.setLayout(self.grid)
        self.openwindows = [False, False]
        self.initUI()
        self.show()

    def initUI(self):
        # Parameters that can be changed
        self.message1 = QLabel("<h3>Parameters</h3>",self)
        self.message1.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.grid.addWidget(self.message1)

        self.message2 = QLabel("Parameter 1",self)
        self.textbox = QLineEdit('20',self)
        self.message2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.textbox.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.grid.addWidget(self.message2)
        self.grid.addWidget(self.textbox)

        self.message3 = QLabel("Parameter 2",self)
        self.textbox2 = QLineEdit('20',self)
        self.message3.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.textbox2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.grid.addWidget(self.message3)
        self.grid.addWidget(self.textbox2)

        # Button to load image
        self.button1 = QPushButton('Load Image Frames',self)
        self.button1.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.grid.addWidget(self.button1, *(10,0))

        # Button interactions
        self.button1.clicked.connect(self.load)

    def load(self):
        self.dialog1 = Slideshow(self)

class Slideshow(QMainWindow):
    def __init__(self, parent = None):
        super(Slideshow, self).__init__(parent)
        self.setWindowTitle("Small Target Tracker")
        self.setGeometry(100,100, 800,800)

        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.pathname = QFileDialog.getExistingDirectory(self, "Select directory")
        
        if self.pathname:
            self.data = Dataloader(self.pathname, img_file_pattern='*.jpg', frame_range=(1, 2))
            self.show()
        else:
            self.close()
            return
            
        self.label = QLabel("",self)
        self.label.resize(800,800)
        self.label.move(0,0)
        self.label.show()

        self.num_object = []
        self.img = []
        for frame, img, gtdata in self.data:
            i = Image.fromarray(img, mode='RGB')
            qt_img = ImageQt.ImageQt(i)
            
            image = QPixmap.fromImage(qt_img)
            
            self.painterInstance = QPainter(image)
            self.penRectangle = QPen(Qt.green)
            self.penRectangle.setWidth(3)

            self.num_object.append(len(gtdata))
            for e in gtdata:
                # draw rectangle on painter
                self.painterInstance.setPen(self.penRectangle)
                self.painterInstance.drawRect(e[0],e[1],e[2],e[3])

            self.img.append(image)
        
        for i in self.img:
            self.label.setPixmap(i)
            self.label.show()

        self.graphData()
        self.dialog2 = Statistics(self)
        self.dialog2.show()
        return

    def graphData(self):
        pen = pg.mkPen(width = 10)
        self.graphWidget = pg.PlotWidget()
        self.graphWidget.show()
        seconds = [i for i in range(1, len(self.num_object)+1)]
        print(self.num_object)
        objects = self.num_object

        self.graphWidget.plot(seconds, objects, pen=pen)
        self.graphWidget.setTitle("Moving objects per frame", size = "20pt")
        self.graphWidget.setLabel('left', 'Average Moving Object (per frame)')
        self.graphWidget.setLabel('bottom', 'Seconds (s)')

class Statistics(QMainWindow):
    def __init__(self, parent = None):
        super(Statistics, self).__init__(parent)
        self.setWindowTitle("Small Target Tracker")
        self.setGeometry(0,0, 300, 210)
        self.initUI()
        self.show()
    
    def initUI(self):
        self.title = QLabel("<h3>Statistics</h3>",self)
        self.title.move(110, 10)

        self.stat1 = QLabel("Unmatch Ground Truth",self)
        self.stat1.adjustSize()
        self.stat1.move(10, 60)
        self.stat1a = QLabel("/Change tracks", self)
        self.stat1a.adjustSize()
        self.stat1a.move(10, 75)

        self.stat2 = QLabel("Average Precision Score",self)
        self.stat2.adjustSize()
        self.stat2.move(10, 110)

        self.stat3 = QLabel("Recall score",self)
        self.stat3.adjustSize()
        self.stat3.move(10, 160)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = Control()
    sys.exit(app.exec())