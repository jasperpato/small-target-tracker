import numpy as np


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
            iou = intersection_over_union(pred, gt_bboxes[i])
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
    

def is_intersecting(box1, box2):
    """
    Check if two boxes are intersecting. Boxes are in the format
    [tl_x, tl_y, width, height]
    """
    
    if box1[0] > box2[0] + box2[2]:
        return False  # box1 is right of box2
    if box2[0] > box1[0] + box1[2]:
        return False  # box1 is left of box2
    if box1[1] > box2[1] + box2[3]:
        return False  # box1 is below box2
    if box2[1] > box1[1] + box1[3]:
        return False  # box1 is above box2
    return True


def intersection_over_union(box1, box2):
    """
    Calculate the intersection over union (IOU) of two boxes.
    Boxes are in the format [tl_x, tl_y, width, height]
    """
    
    xtl = max(box1[0], box2[0])
    ytl = max(box1[1], box2[1])
    xbr = min(box1[0] + box1[2], box2[0] + box2[2])
    ybr = min(box1[1] + box1[3], box1[1] + box2[3])
    
    if is_intersecting(box1, box2):
        intersection = (xbr - xtl) * (ybr - ytl)
        union = box1[2] * box1[3] + box2[2] * box2[3] - intersection
        iou = max(intersection / union, 0)
    else:
        iou = 0.0
    return iou