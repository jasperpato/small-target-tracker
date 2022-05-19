import sys
import argparse
from matplotlib.pyplot import show
import numpy as np
import pyqtgraph as pg

from dataparser import Dataloader
from object_detection import get_thresholds, objects, grow, filter
from kalman_filter import KalmanFilter
from match import association
from evaluation import *

from PySide6.QtCore import Qt, Slot, QBasicTimer, QStringListModel
from PySide6.QtWidgets import QMainWindow, QApplication, QWidget, QPushButton, QLineEdit, QMessageBox, QProgressBar, QLabel, QGridLayout, QSizePolicy, QFileDialog
from PySide6.QtGui import QPixmap, QPainter, QPen
from PIL import Image, ImageQt

from skimage import measure
from skimage import color


parser = argparse.ArgumentParser(description='Find thresholds for cue detection')
parser.add_argument('--dataset_path', type=str, required=True, help='Path to the VISO/mot/car/{frame_no} dataset')
parser.add_argument('--show_blobs', action='store_true', default=False, help='Show blobs instead of tracks')
parser.add_argument('--step', type=int, default=10, help='Interframe difference to use for cue detection')
parser.add_argument('--min_frame', type=int, default=1, help='Minimum frame number to use.')
parser.add_argument('--max_frame', type=int, default=-1, help='Maximum frame number to use. -1 selects up to the highest frame number')

running_windows = []

MAX_FRAME = None
STEP = None


class Slideshow(QMainWindow):
    def __init__(self, dataset_path, parent=None, frame_range=(1, -1)):
        super(Slideshow, self).__init__(parent)
        running_windows.append(self)
        self.initSlideShow()
        self.num_detected = []
        self.images = []
        self.morph_thresholds = get_thresholds()
        self.tracks = []
        self.previous_cues = None
        
        loader = Dataloader(f'{dataset_path}', img_file_pattern='*.jpg', frame_range=frame_range)
        frame_nums = loader.frames
        nframes = len(frame_nums)

        for i in range(step, nframes - step, step):
            self.pbar.setValue(int(round((i - step) / (nframes - step) * 100)))
            QApplication.processEvents()
            
            f0, f1, f2 = [loader(frame_nums[i+j*step]) for j in (-1, 0, 1)]
            img_arr = Image.fromarray(f1[1], mode='RGB')
            image = QPixmap.fromImage(ImageQt.ImageQt(img_arr))
            
            # Main algorithm
            pred_bboxes, gt_bboxes = self.processFrame((f0, f1, f2), is_start_frame=i==step)
            self.num_detected.append(len(pred_bboxes))
            
            # Draw bounding boxes
            self.painterInstance = QPainter(image)
            self.drawBoundingBoxes(pred_bboxes, gt_bboxes)
            image = image.scaled(500, 500, Qt.KeepAspectRatio, Qt.FastTransformation)
            self.images.append(image)
            
            
        self.dialog1.close()
        self.timer = QBasicTimer()
        self.current_frame = 0
        self.delay = 1000
        self.timerEvent()
        
        
    def initSlideShow(self):
        self.label = QLabel(self)
        self.label.resize(500,500)
        self.label.move(50,50)

        self.title = QLabel("<h3>Statistics</h3>",self)
        self.title.move(600, 10)

        self.title1 = QLabel("<h4>Live statistics</h4>", self)
        self.title1.move(600, 30)

        self.title1 = QLabel("<h4>Final statistics</h4>", self)
        self.title1.move(600,125)

        self.stat1 = QLabel("Unmatch Ground Truth : 0.00",self)
        self.stat1.adjustSize()
        self.stat1.move(600, 50)

        self.stat1a = QLabel("Tracks changed : 00", self)
        self.stat1a.adjustSize()
        self.stat1a.move(600, 100)

        self.stat2 = QLabel("Average Precision Score",self)
        self.stat2.adjustSize()
        self.stat2.move(600, 150)

        self.stat3 = QLabel("Average Recall score",self)
        self.stat3.adjustSize()
        self.stat3.move(600, 200)

        self.stat4 = QLabel("Average F1 score",self)
        self.stat4.adjustSize()
        self.stat4.move(600, 250)

        self.button = QPushButton(self)
        self.button.setText("Close All")
        self.button.move(600,300)
        self.button.clicked.connect(self.button_clicked)
        
        self.dialog1 = ProgressBar()
        self.pbar = self.dialog1.pbar
        
    
    def processFrame(self, frames, is_start_frame=False):
        grays = [color.rgb2gray(im) for _, im, _ in frames]
        
        # Candidate small objects detection
        binary = objects(grays)
        
        # Candidate match discrimination
        grown = grow(grays[1], binary, copy = True)
        filtered = filter(grown, self.morph_thresholds, copy = True)

        labeled_image = measure.label(filtered, background=0, connectivity=1)
        blobs = measure.regionprops(labeled_image)
        self.previous_cues = blobs
        
        # Application of kalman filter
        if is_start_frame:
            self.tracks = [KalmanFilter(b, covar = 0.001) for b in blobs]
        else:
            self.tracks = KalmanFilter.assign_detections_to_tracks(
                np.array(blobs), np.array(self.tracks), np.array(self.previous_cues))
        
        pred_bboxes = [cand.bbox for cand in self.tracks]
        gt_bboxes = [Box(gt_box[0], gt_box[1], gt_box[2], gt_box[3]) for gt_box in frames[1][2]]
        return pred_bboxes, gt_bboxes
        
    
    def drawBoundingBoxes(self, pred_bboxes, gt_bboxes):    
        self.penRectangle = QPen(Qt.green)
        self.penRectangle.setWidth(3)
        self.painterInstance.setPen(self.penRectangle)

        # Draw ground truth bounding boxes
        for gt_box in gt_bboxes:
            self.painterInstance.drawRect(gt_box.xtl, gt_box.ytl, gt_box.w, gt_box.h)
        
        self.penRectangle = QPen(Qt.blue)
        self.penRectangle.setWidth(3)
        self.painterInstance.setPen(self.penRectangle)

        # Draw prediction bounding boxes
        for pred_box in pred_bboxes:
            self.painterInstance.drawRect(pred_box.xtl, pred_box.ytl, pred_box.w, pred_box.h)
            
        self.painterInstance.end()
    
    
    def graphTimeSeries(self):
        penPrec = pg.mkPen('r', width=5)
        penRec = pg.mkPen('g', width=5)
        
        plt = pg.plot('Precision and Recall per frame')
        plt.addLegend()

        plt.plot(self.precisions, pen=penPrec, name='precision')
        plt.plot(self.recalls, pen=penRec, name='recall')
        plt.setLabel('left', 'Score')
        plt.setLabel('bottom', 'Frame no.')
        running_windows.append(plt)

        plt = pg.plot('Number of moving objects detected per frame')
        plt.plot(self.num_detected, pen=pg.mkPen(width = 5))
        plt.setLabel('left', 'Num detected')
        plt.setLabel('bottom', 'Frame no.')
        running_windows.append(plt)
        
        plt = pg.plot('Unmatched ground truth proportion per frame')
        plt.plot(1 - self.recalls, pen=pg.mkPen(width = 5))
        plt.setLabel('left', 'Num detected')
        plt.setLabel('bottom', 'Frame no.')
        running_windows.append(plt)
        
        plt = pg.plot('Number of switched tracks per frame')
        plt.plot(1 - self.changed_tracks, pen=pg.mkPen(width = 5))
        plt.setLabel('left', 'Num switched tracks')
        plt.setLabel('bottom', 'Frame no.')
        running_windows.append(plt)
    
    
    def timerEvent(self, e=None):   
        self.timer.start(self.delay, self)
        self.label.setPixmap(self.images[self.current_frame])
        self.label.show()
        self.current_frame = (self.current_frame + 1) % len(self.images)
    
    
    def button_clicked(self):
        global running_windows
        for w in running_windows:
            w.close()
        self.close()


class ProgressBar(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        self.pbar = QProgressBar(self)
        self.pbar.setGeometry(30, 40, 200, 25)
        self.pbar.setValue(0)
        self.setWindowTitle("Launching Tracker")
        self.setGeometry(32,32,320,100)
        self.show()


if __name__ == "__main__":
    args = parser.parse_args()
    dataset_path = args.dataset_path
    show_blobs = args.show_blobs
    step = args.step
    max_frame = args.max_frame
    min_frame = args.min_frame

    app = QApplication(sys.argv)
    widget = Slideshow(dataset_path, frame_range=(min_frame, max_frame))
    widget.resize(1000,600)
    widget.setWindowTitle("Small target tracker")
    widget.show()
    
    sys.exit(app.exec())