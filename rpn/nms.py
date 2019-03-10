import numpy as np

def nms(boxes, nms_thresh=0.8):
    """Pure Python NMS baseline."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, -1]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thresh)[0]
        order = order[inds + 1]

    return keep

def nms_with_limit(boxes, nms_thresh=0.8):
    if len(boxes) == 0:
        return []
    pick = []
        
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    
    idxs = np.argsort(boxes[:,-1])
    
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        area_i = np.maximum(0, x2[i] - x1[i] + 1) * np.maximum(0, y2[i] - y1[i] + 1)
        area_array = np.zeros(len(idxs) - 1)
        area_array.fill(area_i)
        overlap = (w * h) / (area_array)
        idxs = np.delete(idxs, np.concatenate(([last],np.where(overlap > nms_thresh)[0])))
    return pick

def nms_vote(boxes, nms_thresh=0.8):
    if len(boxes)==0:
        return []
    pick_boxes=[]
    xmin=boxes[:,0]
    ymin=boxes[:,1]
    xmax=boxes[:,2]
    ymax=boxes[:,3]
    
    areas=(xmax-xmin+1)*(ymax-ymin+1)
    
    orders=np.argsort(boxes[:,-1])[::-1]
    
    idx=0
    while orders.size>0:
        index=orders[idx]
        
        x1=np.maximum(xmin[index], xmin[orders[idx+1:]])
        y1=np.maximum(ymin[index], ymin[orders[idx+1:]])
        x2=np.minimum(xmax[index], xmax[orders[idx+1:]])
        y2=np.minimum(ymax[index], ymax[orders[idx+1:]])
        
        w = np.maximum(0, x2 - x1 + 1)
        h = np.maximum(0, y2 - y1 + 1)
        
        inter=w*h
        
        overlap=inter/(areas[index]+areas[orders[idx+1:]]-inter)
        
        '''where on orders'''
        inds=np.where(overlap<=nms_thresh)[0]
        
        vote_inds=np.where(overlap>nms_thresh)[0]
        vote_inds=np.append(orders[vote_inds+1], index)
        
        vote_box=np.mean(boxes[vote_inds], axis=0)
        
        pick_boxes.append({'index':index, 'box':vote_box})
        orders=orders[inds+1]
        
    pick_boxes=sorted(pick_boxes, key=lambda x:x['index'])
    return pick_boxes