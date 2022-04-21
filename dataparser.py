from glob import glob
import cv2
import psutil
import regex as re
import pandas as pd


class Dataloader(object):
    def __init__(self, folder_path, img_file_pattern='*.jpg', gt_file_pattern="gt.txt", frame_range=None):        
        img_paths = glob(f'{folder_path}/**/{img_file_pattern}', recursive=True)
        gt_file = glob(f'{folder_path}/**/{gt_file_pattern}', recursive=True)[0]
        
        get_frame = lambda p: int(re.search('\d+', p).group())
        self.frame_paths = {get_frame(p): p for p in img_paths}
        frames = self.frame_paths.keys()
        
        self.frame_range = frame_range if frame_range is not None else (min(frames), max(frames))
        self.gt_data = pd.read_csv(gt_file, sep=',', names=['frame', 'track_id', 'tl_x', 'tl_y', 'width', 'height'])
        self.preloaded_frames = {}
        self.preload_frames()
        

    def preload_frames(self):
        start_frame, end_frame = self.frame_range
        loaded_frames = self.loaded_frames.keys()
        frame_no = max(loaded_frames) + 1 if loaded_frames else start_frame
        
        while psutil.virtual_memory().percent > 90 and frame_no < end_frame:
            img = cv2.imread(self.frame_paths[frame_no], cv2.IMREAD_UNCHANGED)
            gt_data = self.gt_data[self.gt_data.frame.eq(frame_no)]
            gt_data = gt_data[['tl_x', 'tl_y', 'width', 'height']]  
            self.loaded_frames[frame_no] = (frame_no, img, gt_data.to_numpy())
            frame_no += 1
    
    
    def get_full_frame_range(self):
        frames = self.frame_paths.keys()
        return (min(frames), max(frames))
      

    def __call__(self, frame_no):
        if frame_no in self.loaded_frames:
            return self.loaded_frames.pop(frame_no)
        else:
            img = cv2.imread(self.frame_paths[frame_no], cv2.IMREAD_UNCHANGED)
            gt_data = self.gt_data[self.gt_data.frame.eq(frame_no)]
            gt_data = gt_data[['tl_x', 'tl_y', 'width', 'height']]  
            return (frame_no, img, gt_data.to_numpy())
        
    
    def __iter__(self):
        self.current_frame = self.frame_range[0]
        return self
            

    def __next__(self):
        if self.current_frame > self.frame_range[1]:
            raise StopIteration
        else:
            data = self.loaded_frames.pop(self.current_frame, None) or self.__call__(self.current_frame)
            self.current_frame += 1
            return data
    
    
    