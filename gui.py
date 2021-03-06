import sys
import argparse
import numpy as np
import pyqtgraph as pg

from dataparser import Dataloader
from object_detection import get_thresholds, objects, grow, filter
from kalman_filter import KalmanFilter
from evaluation import *

from PySide6.QtCore import Qt, QBasicTimer
from PySide6.QtWidgets import QMainWindow, QApplication, QWidget, QPushButton, QProgressBar, QLabel, QGridLayout, QLineEdit
from PySide6.QtGui import QPixmap, QPainter, QPen
from PIL import Image, ImageQt

from skimage import measure
from skimage import color


parser = argparse.ArgumentParser(description='Find thresholds for cue detection')
parser.add_argument('--dataset_path', type=str, required=True, help='Path to the VISO/mot/car/{frame_no} dataset')
parser.add_argument('--min_frame', type=int, default=1, help='Minimum frame number to use.')
parser.add_argument('--max_frame', type=int, default=-1, help='Maximum frame number to use. -1 selects up to the highest frame number')

running_windows = []

MAX_FRAME = None
STEP = None


class Slideshow(QMainWindow):
    def __init__(self, dataset_path, frame_diff, h, frame_range=(1, -1), parent=None):
        super(Slideshow, self).__init__(parent) 
        #format of h : [covar, cost, m.thresh,m.rad,obj.thresh,region.thresh,thresh.diff]
        self.h = h
        running_windows.append(self)
        self.initSlideShow()
        self.images = []
        self.morph_thresholds = get_thresholds()
        
        self.num_detected = []
        self.precisions = []
        self.recalls = []
        self.f1 = []
        self.num_changed_tracks = []
        
        self.tracks = []
        self.previous_cues = None
        self.kalman2gt = {}
        
        loader = Dataloader(f'{dataset_path}', img_file_pattern='*.jpg', frame_range=frame_range)
        # Had to retrieve correct number of frames so that GTboxes work
        frame_nums = loader.frame_nums
        for i in range(frame_diff, len(frame_nums) - frame_diff):
            progress_prop = (i - frame_diff) / (len(frame_nums) - 2 * frame_diff)
            self.pbar.setValue(int(progress_prop * 100))
            QApplication.processEvents()
            
            f0, f1, f2 = [loader(frame_nums[i+j*frame_diff]) for j in (-1, 0, 1)]
            img_arr = Image.fromarray(f1[0], mode='RGB')
            image = QPixmap.fromImage(ImageQt.ImageQt(img_arr))
            
            # Main algorithm
            pred_bboxes, gt_bboxes = self.processFrame((f0, f1, f2), is_start_frame=i==frame_diff)
            self.num_detected.append(len(pred_bboxes))
            
            # Record stats
            m = evaluation_metrics(np.array(pred_bboxes), np.array(gt_bboxes))
            self.precisions.append(m['precision'])
            self.recalls.append(m['recall'])
            self.f1.append(m['f1'])
            
            # Calculate number of changed tracks
            detected_gts = m['pred2gt']
            kalman2ind = dict(zip(self.tracks, detected_gts))
            kalman2gt = {k: v for k, v in kalman2ind.items() if v != -1}
            num_changed_tracks = sum(kalman2gt[k] != self.kalman2gt[k] for k in kalman2gt if k in self.kalman2gt)
            self.num_changed_tracks.append(num_changed_tracks)
            self.kalman2gt = kalman2gt

            # Draw bounding boxes
            self.painterInstance = QPainter(image)
            self.drawBoundingBoxes(pred_bboxes, gt_bboxes)
            image = image.scaled(700, 700, Qt.KeepAspectRatio, Qt.FastTransformation)
            self.images.append(image)
        
        self.dialog1.close()

        # Update UI with metrics
        self.num_detected = np.array(self.num_detected)
        self.precisions = np.array(self.precisions)
        self.recalls = np.array(self.recalls)
        self.f1 = np.array(self.f1)
        self.num_changed_tracks = np.array(self.num_changed_tracks)
        self.stat2.setText("Average Precision Score : {}".format(np.average(self.precisions)))
        self.stat3.setText("Average Recall score: {}".format(np.average(self.recalls)))
        self.stat4.setText("Average F1 score : {}".format(np.average(self.f1)))
        self.graphTimeSeries()
        self.timer = QBasicTimer()
        self.current_frame = 0
        self.delay = 1000
        self.timerEvent()
        
        
    def initSlideShow(self):
        self.label = QLabel(self)
        self.label.resize(700,700)
        self.label.move(50,50)

        self.title = QLabel("<h2>Statistics</h2>",self)
        self.title.move(760, 10)

        self.stat2 = QLabel("Average Precision Score : 00.00",self)
        self.stat2.adjustSize()
        self.stat2.move(760, 50)

        self.stat3 = QLabel("Average Recall score: 00.00",self)
        self.stat3.adjustSize()
        self.stat3.move(760, 100)

        self.stat4 = QLabel("Average F1 score : 00.00",self)
        self.stat4.adjustSize()
        self.stat4.move(760, 150)

        self.button = QPushButton(self)
        self.button.setText("Close All")
        self.button.move(760,200)
        self.button.clicked.connect(self.button_clicked)
        
        self.dialog1 = ProgressBar()
        self.pbar = self.dialog1.pbar
        
    
    def processFrame(self, frames, is_start_frame=False):
        ctr_frame = frames[1]
        grays = [color.rgb2gray(im) for im, _ in frames]
        
        # Candidate small objects detection
        binary = objects(grays, thresh=self.h[4])

        # Candidate match discrimination
        grown = grow(grays[1], binary, thresh=self.h[5], diff_thresh=self.h[6], copy=True)
        filtered = filter(grown, self.morph_thresholds, copy = True)

        # CCA to find hypotheses
        labeled_image = measure.label(filtered, background=0, connectivity=1)
        blobs = measure.regionprops(labeled_image)
        self.previous_cues = blobs
        
        # Application of kalman filter
        if is_start_frame:
            self.tracks = [KalmanFilter(b, covar=self.h[0]) for b in blobs]
        else:
            self.tracks = KalmanFilter.assign_detections_to_tracks(
                np.array(blobs), np.array(self.tracks), np.array(self.previous_cues),
                covar=self.h[0], pcost=self.h[1], ssim_thresh=self.h[2], ssim_rad=self.h[3])
        
        pred_bboxes = [cand.bbox for cand in self.tracks]
        gt_bboxes = [Box(gt_box[0], gt_box[1], gt_box[2], gt_box[3]) for gt_box in ctr_frame[1]]
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
        plt = pg.plot()
        plt.addLegend()

        plt.plot(self.precisions, pen=penPrec, name='precision')
        plt.plot(self.recalls, pen=penRec, name='recall')
        plt.setLabel('left', 'Score')
        plt.setLabel('bottom', 'Frame no.')
        plt.setWindowTitle('Precision and Recall per frame')
        running_windows.append(plt)

        plt = pg.plot()
        plt.plot(self.num_detected, pen=pg.mkPen(width = 5))
        plt.setLabel('left', 'Num detected')
        plt.setLabel('bottom', 'Frame no.')
        plt.setWindowTitle('Number of moving objects detected per frame')
        running_windows.append(plt)
        
        plt = pg.plot()
        plt.plot(1 - self.recalls, pen=pg.mkPen(width = 5))
        plt.setLabel('left', 'Num detected')
        plt.setLabel('bottom', 'Frame no.')
        plt.setWindowTitle('Unmatched ground truth proportion per frame')
        running_windows.append(plt)
        
        plt = pg.plot()
        plt.plot(1 - self.num_changed_tracks, pen=pg.mkPen(width = 5))
        plt.setLabel('left', 'Num switched tracks')
        plt.setLabel('bottom', 'Frame no.')
        plt.setWindowTitle('Number of switched tracks per frame')
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
        self.pbar.setGeometry(30, 40, 250, 25)
        self.pbar.setValue(0)
        self.setWindowTitle("Launching Tracker")
        self.setGeometry(32,32,320,100)
        self.show()


class Start(QWidget):
    def __init__(self,parent = None):
        super().__init__()
        self.layout = QGridLayout()
        self.setLayout(self.layout)
        
        self.layout.addWidget(QLabel("Kalman Covariance Factor"), 0, 0, 1,1)
        self.lineEdit1 = QLineEdit()
        self.lineEdit1.setText("0.001")
        self.layout.addWidget(self.lineEdit1, 0, 1, 1,1)

        self.layout.addWidget(QLabel("Unassociated Track/Hypothesis Cost"), 0, 2, 1,1)
        self.lineEdit2 = QLineEdit()
        self.lineEdit2.setText("10")
        self.layout.addWidget(self.lineEdit2, 0, 3, 1,1)

        self.layout.addWidget(QLabel("Minimum SSIM for Nearest Hypothesis Match"), 1, 0, 1,1)
        self.lineEdit3 = QLineEdit()
        self.lineEdit3.setText("0.8")
        self.layout.addWidget(self.lineEdit3, 1, 1, 1,1)

        self.layout.addWidget(QLabel("Maximum Radius for Nearest Hypothesis Match"), 1, 2, 1,1)
        self.lineEdit4 = QLineEdit()
        self.lineEdit4.setText("5")
        self.layout.addWidget(self.lineEdit4, 1, 3, 1,1)

        self.layout.addWidget(QLabel("Candidate Detection Probability Threshold"), 2, 0, 1,1)
        self.lineEdit5 = QLineEdit()
        self.lineEdit5.setText("0.05")
        self.layout.addWidget(self.lineEdit5, 2, 1, 1,1)

        self.layout.addWidget(QLabel("Region Growing Probability Threshold"), 2, 2, 1,1)
        self.lineEdit6 = QLineEdit()
        self.lineEdit6.setText("0.005")
        self.layout.addWidget(self.lineEdit6, 2, 3, 1,1)

        self.layout.addWidget(QLabel("Interframe difference for Candidate Detection"), 3, 0, 1,1)
        self.lineEdit7 = QLineEdit()
        self.lineEdit7.setText("10")
        self.layout.addWidget(self.lineEdit7, 3, 1, 1,1)

        self.layout.addWidget(QLabel("Maximum Threshold Difference for Region Growing"), 3, 2, 1,1)
        self.lineEdit8 = QLineEdit()
        self.lineEdit8.setText("0.5")
        self.layout.addWidget(self.lineEdit8, 3, 3, 1,1)

        self.button = QPushButton("Start")
        self.button.clicked.connect(self.start)
        self.layout.addWidget(self.button, 4, 3, 1,1)


    def start(self):
        hyper_param = []
        hyper_param.append(float(self.lineEdit1.text()))
        hyper_param.append(float(self.lineEdit2.text()))
        hyper_param.append(float(self.lineEdit3.text()))
        hyper_param.append(float(self.lineEdit4.text()))
        hyper_param.append(float(self.lineEdit5.text()))
        hyper_param.append(float(self.lineEdit6.text()))
        hyper_param.append(float(self.lineEdit8.text()))
        self.close()
        widget = Slideshow(args.dataset_path, int(self.lineEdit7.text()), hyper_param,
                       frame_range=(args.min_frame, args.max_frame))
        widget.resize(1000,800)
        widget.setWindowTitle("Small target tracker")
        widget.show()
        return
    
    
if __name__ == "__main__":
    args = parser.parse_args()
    app = QApplication(sys.argv)
    widget = Start()
    widget.show()
    sys.exit(app.exec())