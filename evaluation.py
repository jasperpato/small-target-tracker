import numpy as np

class Box:
    def __init__(self, xtl, ytl, w, h):
        self.xtl = xtl
        self.ytl = ytl
        self.xbr = w + xtl
        self.ybr = h + ytl
        self.w = w
        self.h = h
    
    @property    
    def area(self):
        return (self.xbr - self.xtl) * (self.ybr - self.ytl)
    
    
def is_intersecting(cls, box1: 'Box', box2: 'Box'):
    if box1.xtl > box2.xbr:
        return False  # box1 is right of box2
    if box2.xtl > box1.xbr:
        return False  # box1 is left of box2
    if box1.ybr < box2.ytl:
        return False  # box1 is above box2
    if box1.ytl > box2.ybr:
        return False  # box1 is below box2
    return True


def intersection_over_union(cls, box1: 'Box', box2: 'Box'):
    """
    Calculate the intersection over union (IOU) of two boxes.
    Boxes are in the format [tl_x, tl_y, width, height]
    """
    xtl = max(box1.xtl, box2.xtl)
    ytl = max(box1.ytl, box2.ytl)
    xbr = min(box1.xbr, box2.xbr)
    ybr = min(box1.ybr, box2.ybr)
    
    if not Box.is_intersecting(box1, box2):
        return 0.0

    intersection = (xbr - xtl) * (ybr - ytl)
    union = box1.area + box2.area - intersection
    return max(intersection / union, 0)
    
    
def evaluation_metrics(pred_bboxes, gt_bboxes, iou_threshold=0.7):
    """ 
    Compute evaluation metrics for a set of predicted and ground truth bounding boxes.
    Args: 
        pred_bboxes: numpy array containing prediction boxes in the format [tl_x, tl_y, width, height]
        gt_bboxes: numpy array containing ground truth boxes in the format [tl_x, tl_y, width, height]
        iou_threshold: threshold for considering a box a positive detection 
    Returns: 
        A dictionary containing the following metrics: precision, recall, f1-score
    """
    
    npos = np.shape(gt_bboxes)[0]
    seen_gts = np.zeros(npos)
    tp = 0
    fp = 0
    
    for pred in pred_bboxes:
        max_iou = 0.0
        for i in range(npos):
            iou = Box.intersection_over_union(pred, gt_bboxes[i])
            if iou > max_iou:
                max_iou = iou
                gt_idx = i  # index of the ground truth box with the highest IoU
        
        if max_iou >= iou_threshold:
            if seen_gts[gt_idx] == 0:   # first time this gt is detected
                tp += 1
                seen_gts[gt_idx] = 1    # mark as detected
            else:
                # duplicate detection
                fp += 1
        else:
            fp += 1

    precision = tp / (tp + fp)
    recall = tp / npos
    f1 = 2 * precision * recall / (precision + recall)
    return {'precision': precision, 'recall': tp / npos, 'f1': f1}
