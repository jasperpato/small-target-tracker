import sys
import time

from dataparser import Dataloader
from evaluation import evaluation_metrics
from morph_thresholds import morph_cues
from object_detection import objects, region_growing

from os import listdir
from os.path import isfile, join
from PySide6.QtCore import Qt, Slot, QBasicTimer, QStringListModel
from PySide6.QtWidgets import QMainWindow, QApplication, QWidget, QPushButton, QLineEdit, QMessageBox, QLabel, QGridLayout, QSizePolicy, QFileDialog
from PySide6.QtGui import QPixmap, QPainter, QPen
from PIL import Image, ImageQt
import pyqtgraph as pg
from skimage import measure, color
running_window = []
'''
Launch the gui.py and click on load image to choose file path.
Select where the mot/car/001 file is.
'''
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

        self.message2 = QLabel("Area",self)
        self.Alow = QLineEdit('20',self)
        self.Ahigh = QLineEdit('20',self)
        self.message2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.Alow.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.Ahigh.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.grid.addWidget(self.message2, 1, 0)
        self.grid.addWidget(self.Alow, 2,0)
        self.grid.addWidget(QLabel("< area <", self), 2,1)
        self.grid.addWidget(self.Ahigh, 2,2)

        self.message3 = QLabel("Extent",self)
        self.Elow = QLineEdit('20',self)
        self.Ehigh = QLineEdit('20',self)
        self.message3.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.Elow.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.Ehigh.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.grid.addWidget(self.message3, 3, 0)
        self.grid.addWidget(self.Elow, 4,0)
        self.grid.addWidget(QLabel("< extent <", self), 4,1)
        self.grid.addWidget(self.Ehigh, 4,2)

        self.message4 = QLabel("Major Axis",self)
        self.Mlow = QLineEdit('20',self)
        self.Mhigh = QLineEdit('20',self)
        self.message4.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.Mlow.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.Mhigh.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.grid.addWidget(self.message4, 5, 0)
        self.grid.addWidget(self.Mlow, 6,0)
        self.grid.addWidget(QLabel("< major axis <", self), 6,1)
        self.grid.addWidget(self.Mhigh, 6,2)

        self.message5 = QLabel("Eccentricity",self)
        self.Eclow = QLineEdit('20',self)
        self.Echigh = QLineEdit('20',self)
        self.message5.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.Eclow.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.Echigh.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.grid.addWidget(self.message5, 7, 0)
        self.grid.addWidget(self.Eclow, 8,0)
        self.grid.addWidget(QLabel("< eccentricity <", self), 8,1)
        self.grid.addWidget(self.Echigh, 8,2)

        # Button to load image
        self.button1 = QPushButton('Load Image Frames',self)
        self.button1.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.grid.addWidget(self.button1, 9,0)

        # Button to close all windows
        self.button2 = QPushButton('Close Windows',self)
        self.button2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.grid.addWidget(self.button2, 9,2)

        # Button interactions
        self.button1.clicked.connect(self.load)
        self.button2.clicked.connect(self.closeall)

    def load(self):
        self.dialog1 = Slideshow(self)

    def closeall(self):
        global running_window
        for i in running_window:
            i.close()
        running_window = []

class Slideshow(QMainWindow):
    def __init__(self, parent = None):
        super(Slideshow, self).__init__(parent)
        global running_window
        self.setWindowTitle("Small Target Tracker")
        self.setGeometry(100,100, 800,800)

        running_window.append(self)

        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.pathname = QFileDialog.getExistingDirectory(self, "Select directory")
        
        if self.pathname:
            loader = Dataloader(self.pathname, img_file_pattern='*.jpg', frame_range=(1, 100))
            pre_frames = list(loader.preloaded_frames.values())
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

        step = 5

        for i in range(step-1, len(pre_frames)-step+1, step):
            print('Progress of calculation: {:d}'.format(i))
            grays = [ color.rgb2gray(f[1]) for f in (pre_frames[i-step+1], pre_frames[i], pre_frames[i+step-1]) ]
            picture = [f[1] for f in (pre_frames[i-step+1], pre_frames[i], pre_frames[i+step-1])]
            binary = objects(grays)
            grown = region_growing(grays[1], binary)
            ncands, ar_avg, ar_std, ex_avg, ex_std, al_avg, al_std, ec_avg, ec_std = morph_cues(binary, pre_frames[i][2], 0.4)

            img = Image.fromarray(picture[0], mode='RGB')
            qt_img = ImageQt.ImageQt(img)
            
            image = QPixmap.fromImage(qt_img)

            self.painterInstance = QPainter(image)
            self.penRectangle = QPen(Qt.green)
            self.penRectangle.setWidth(3)

            self.num_object.append(ncands)
            for box in pre_frames[i][2]:
                # draw rectangle on painter
                self.painterInstance.setPen(self.penRectangle)
                self.painterInstance.drawRect(box[0],box[1],box[2],box[3])

            self.img.append(image)
        print("Finished calculations")
        self.timer = QBasicTimer()
        self.step = 0
        self.delay = 1000 #ms

        self.timerEvent()

    def graphObjectDetected(self):
        global running_window

        pen = pg.mkPen(width = 5)
        self.graphWidget = pg.plot()
        self.graphWidget.show()
        seconds = [i for i in range(1, len(self.num_object)+1)]
        objects = self.num_object

        self.graphWidget.plot(seconds, objects, pen=pen)
        self.graphWidget.setTitle("Moving objects per frame", size = "20pt")
        self.graphWidget.setLabel('left', 'Average Moving Object (per frame)')
        self.graphWidget.setLabel('bottom', 'Seconds (s)')

        running_window.append(self.graphWidget)
    
    def graphScores(self):
        global running_window

        pen1 = pg.mkPen('r',width = 5)
        pen2 = pg.mkPen('g',width = 5)
        self.graphWidget = pg.plot()
        self.graphWidget.addLegend()
        self.graphWidget.show()
        seconds = [1,2,3]
        precision = [1.1,2.1,3.3]
        recall = [2,3,4]

        self.graphWidget.plot(seconds, precision, pen=pen1, name = 'precision')
        self.graphWidget.plot(seconds, recall, pen=pen2, name = 'recall')
        self.graphWidget.setTitle("Precision and Recall score", size = "20pt")
        self.graphWidget.setLabel('left', 'Score')
        self.graphWidget.setLabel('bottom', 'Seconds (s)')

        running_window.append(self.graphWidget)
    
    def timerEvent(self, e=None):
        if self.step >= len(self.img):
            self.timer.stop()
            self.step = 0
            self.painterInstance.end()
            self.graphObjectDetected()
            self.graphScores()
            self.dialog2 = Statistics(self)
            self.dialog2.show()
            return 
        self.timer.start(self.delay, self)
        self.label.setPixmap(self.img[self.step])
        self.label.show()
        self.step += 1

class Statistics(QMainWindow):
    def __init__(self, parent = None):
        super(Statistics, self).__init__(parent)
        global running_window
        self.setWindowTitle("Small Target Tracker")
        self.setGeometry(0,0, 300, 210)
        self.initUI()
        self.show()
        running_window.append(self)
        
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