from glob import glob
from pathlib import Path
from skimage import io
import sys
import cv2
import psutil
import regex as re
import pandas as pd
import matplotlib.pyplot as plt

RAM_LIMIT = 90 # stop loading when RAM reaches this percentage usage


def read_img(frame_no, img_path, gt_df):
    """
    Read image from file path
    """
    img = io.imread(img_path)
    gt_data = gt_df[gt_df.frame.eq(frame_no)]
    gt_data = gt_data[['tl_x', 'tl_y', 'width', 'height']]  
    return frame_no, img, gt_data.to_numpy()


class Dataloader(object):
    def __init__(self, path, img_file_pattern='*.jpg', frame_range=None):        
        img_paths = glob(f'{path}/img/{img_file_pattern}', recursive=True)
        gt_file = glob(f'{path}/gt/gt.txt', recursive=True)[0]
        
        get_frame = lambda p: int(re.search('\d+', Path(p).stem).group())
        self.frame_paths = {get_frame(p): p for p in img_paths}
        
        full_frame_range = self.get_full_frame_range()
        self.frame_range = frame_range or full_frame_range
        if self.frame_range[0] < full_frame_range[0] or self.frame_range[1] > full_frame_range[1]:
            raise ValueError('Frame range {} is out of range {}'.format(frame_range, full_frame_range))
        
        self.gt_df = pd.read_csv(gt_file, sep=',', usecols=range(6), names=['frame', 'track_id', 'tl_x', 'tl_y', 'width', 'height'])
        self.preloaded_frames = {}
        self.preload_frames()
        

    def preload_frames(self):
        start_frame, end_frame = self.frame_range
        frame_no = max(self.preloaded_frames.keys()) + 1 if self.preloaded_frames.keys() else start_frame
        
        while psutil.virtual_memory().percent > 100 - RAM_LIMIT and frame_no < end_frame:
            if frame_no in self.frame_paths:
                img_path = self.frame_paths[frame_no]
                return read_img(frame_no, img_path, self.gt_df)
            frame_no += 1
    
    
    def get_full_frame_range(self):
        frames = self.frame_paths.keys()
        return (min(frames), max(frames))
      

    def __call__(self, frame_no):
        if frame_no in self.preloaded_frames:
            return self.preloaded_frames.get(frame_no)
        elif frame_no in self.frame_paths:
            img_path = self.frame_paths[frame_no]
            return read_img(frame_no, img_path, self.gt_df)
        else:
            raise ValueError('Frame {} could not be found'.format(frame_no))
        
    
    def __iter__(self):
        frames = [f for f in self.frame_paths.keys() 
                  if f >= self.frame_range[0] and f <= self.frame_range[1]]
        self.frame_iterator = iter(sorted(frames))
        return self
            

    def __next__(self):
        frame_no = next(self.frame_iterator)
        return self.__call__(frame_no)
        
    
if __name__ == '__main__':
    dataset_path = sys.argv[1].strip('/')

    # TESTING
    dataloader = Dataloader(f'{dataset_path}/car/001', img_file_pattern='*.jpg', frame_range=(1, 100))
    frames = list(dataloader.preloaded_frames.values())[:3]
    
    
    