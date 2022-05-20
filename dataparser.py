from glob import glob
from pathlib import Path
from typing import Sequence
from skimage import io
import sys
import psutil
import regex as re
import pandas as pd


RAM_LIMIT = 90 # stop loading when RAM reaches this percentage usage


def read_img(frame_no, img_path, gt_df):
    """
    Read image from file path
    """
    img = io.imread(img_path)
    gt_data = gt_df[gt_df.frame.eq(frame_no)]
    gt_data = gt_data[['tl_x', 'tl_y', 'width', 'height']]  
    return img, gt_data.to_numpy()


class Dataloader(object):
    def __init__(self, path, img_file_pattern='*.jpg', frame_range=None, preload_frames=True):        
        img_paths = glob(f'{path}/img/{img_file_pattern}', recursive=True)
        gt_file = glob(f'{path}/gt/gt.txt', recursive=True)[0]
        
        get_frame = lambda p: int(re.search('\d+', Path(p).stem).group())
        self.frame_paths = {get_frame(p): p for p in img_paths}
        self.set_frame_range(frame_range)
        
        with open(gt_file) as f:
            l1 = f.readline()
            sep = ',' if ',' in l1 else ' '
            gt_df = pd.read_csv(gt_file, sep=sep, usecols=range(6), 
                                    names=['frame', 'track_id', 'tl_x', 'tl_y', 'width', 'height'])
            self.gt_df = gt_df[gt_df.frame.between(*self.frame_range)]
            
        self.preloaded_frames = {}
        if preload_frames:
            frame_no = self.frame_range[0]
            while psutil.virtual_memory().percent > 100 - RAM_LIMIT and frame_no <= self.frame_range[1]:
                if frame_no in self.frame_paths:
                    img_path = self.frame_paths[frame_no]
                    self.preloaded_frames[frame_no] = read_img(frame_no, img_path, self.gt_df)
                frame_no += 1
        
    @property
    def frames(self):
        all_frames = sorted(self.frame_paths.keys())
        return [f for f in all_frames if f >= self.frame_range[0] and f <= self.frame_range[1]]
    
    def set_frame_range(self, input_range=None):
        all_frames = self.frame_paths.keys()
        full_range = (min(all_frames), max(all_frames))
        if input_range is None:
            self.frame_range = full_range
        else:
            assert isinstance(input_range, Sequence) and len(input_range) == 2, \
                f'Frame range should be a sequence (e.g., tuple) of length 2'
            assert input_range[0] >= full_range[0] and input_range[1] <= full_range[1], \
                f'Provided frame range {input_range} is out of range of {full_range}'
            assert input_range[0] <= input_range[1], 'Provided frame range is invalid'
            assert input_range[0] > 0 and input_range[1] > 0, 'Frame range values should be positive'
            self.frame_range = input_range
              
    def __call__(self, frame_no):
        if frame_no in self.preloaded_frames:
            return self.preloaded_frames.get(frame_no)
        elif frame_no in self.frame_paths:
            img_path = self.frame_paths[frame_no]
            return read_img(frame_no, img_path, self.gt_df)
        else:
            raise ValueError('Frame {} could not be found'.format(frame_no))
        
    def __iter__(self):
        frames = [f for f in self.frame_paths.keys() if f >= self.frame_range[0] and f <= self.frame_range[1]]
        self.frame_iterator = iter(sorted(frames))
        return self
            
    def __next__(self):
        frame_no = next(self.frame_iterator)
        return self.__call__(frame_no)
        
    
if __name__ == '__main__':
    dataset_path = sys.argv[1].rstrip('/')

    # TESTING
    dataloader = Dataloader(f'{dataset_path}/car/028', img_file_pattern='*.jpg', frame_range=(1, 100))
    print('Frames', len(dataloader.frames))
    
    
    