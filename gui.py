import sys
import numpy as np

from dataparser import Dataloader
from object_detection import get_thresholds, objects, grow, filter

from PySide6.QtCore import Qt, Slot, QBasicTimer, QStringListModel
from PySide6.QtWidgets import QMainWindow, QApplication, QWidget, QPushButton, QLineEdit, QMessageBox, QLabel, QGridLayout, QSizePolicy, QFileDialog
from PySide6.QtGui import QPixmap, QPainter, QPen
from PIL import Image, ImageQt
import pyqtgraph as pg
from skimage import measure, color

from numpy import maximum
from skimage import measure

running_window = []
dataset_path = None
max_frame = None
step = None

'''
python3 gui.py {....../car/{Pick a number}} {Number of images to load} {Steps between frames}
'''


class Slideshow(QWidget):
    def __init__(self, parent = None):
        super(Slideshow, self).__init__(parent)
        global running_window
        global dataset_path
        global max_frame
        global step


        running_window.append(self)
        
        loader = Dataloader(f'{dataset_path}', img_file_pattern='*.jpg', frame_range=(1, max_frame))
        pre_frames = list(loader.preloaded_frames.values())
            
        self.label = QLabel(self)
        self.label.resize(500,500)
        self.label.move(50,50)
        self.label.show()

        self.num_object = []
        self.img = []

        thresholds = get_thresholds()

        for i in range(step, len(pre_frames)-step, step):
            print('Progress of calculation: {:d}'.format(i))

            picture = [f[1] for f in (pre_frames[i-step], pre_frames[i], pre_frames[i+step])]
            grays = [ color.rgb2gray(p) for p in picture ]
                
            binary = objects(grays)
            grown = grow(grays[1], binary)
            filtered = filter(grown, thresholds)

            ncands = np.amax(measure.label(filtered))
            print(ncands)

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
            
            self.painterInstance.end()
            image = image.scaled(500, 500, Qt.KeepAspectRatio, Qt.FastTransformation)
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

    dataset_path = sys.argv[1].rstrip('/')
    max_frame = int(sys.argv[2])
    step = int(sys.argv[3])

    app = QApplication(sys.argv)
    widget = Slideshow()
    widget.resize(1000,1000)
    widget.setWindowTitle("Small target tracker")
    widget.show()
    sys.exit(app.exec())