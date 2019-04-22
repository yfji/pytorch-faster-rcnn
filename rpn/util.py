import numpy as np

DEBUG=True

def bbox_overlaps(b1, b2):
    ws1=b1[:,2]-b1[:,0]+1
    hs1=b1[:,3]-b1[:,1]+1
    ws2=b2[:,2]-b2[:,0]+1
    hs2=b2[:,3]-b2[:,1]+1

    area1=ws1*hs1
    area2=ws2*hs2
    bbox_overlap=np.zeros((b1.shape[0],b2.shape[0]))
    
    if b1.shape[0]<=b2.shape[0]:
        for i, bbox in enumerate(b1):
            x1=np.maximum(bbox[0], b2[:,0])
            y1=np.maximum(bbox[1], b2[:,1])
            x2=np.minimum(bbox[2], b2[:,2])
            y2=np.minimum(bbox[3], b2[:,3])

            w=np.maximum(0, x2-x1+1)
            h=np.maximum(0, y2-y1+1)

            areas=w*h
            iou=areas/(area1[i]+area2-areas)

            bbox_overlap[i,:]=iou[...]
    else:
        for i, bbox in enumerate(b2):
            x1=np.maximum(bbox[0], b1[:,0])
            y1=np.maximum(bbox[1], b1[:,1])
            x2=np.minimum(bbox[2], b1[:,2])
            y2=np.minimum(bbox[3], b1[:,3])

            w=np.maximum(0, x2-x1+1)
            h=np.maximum(0, y2-y1+1)

            areas=w*h
            iou=areas/(area2[i]+area1-areas)
            bbox_overlap[:,i]=iou[...]
            

    return bbox_overlap


def bbox_overlaps_per_image(anchors, boxes):
    return bbox_overlaps(anchors, boxes)
 
def bbox_overlaps_batch(anchors, boxes, batch_size):
    bbox_overlaps_batch=[]
    for i in range(batch_size):
        bbox_overlaps=bbox_overlaps_per_image(anchors, boxes[i]) 
        bbox_overlaps_batch.append(bbox_overlaps)
    return bbox_overlaps_batch

def bbox_transform(rois, gt_rois):
#    print(gt_box)
    ws=rois[:,2]-rois[:,0]+1
    hs=rois[:,3]-rois[:,1]+1
    
    ws[ws<0]=0
    hs[hs<0]=0
    
    assert(np.all(ws))
    assert(np.all(hs))
        
    ws_gt=gt_rois[:,2]-gt_rois[:,0]+1
    hs_gt=gt_rois[:,3]-gt_rois[:,1]+1
    
    ctrx=rois[:,0]+ws*0.5
    ctry=rois[:,1]+hs*0.5
    gt_ctrx=gt_rois[:,0]+ws_gt*0.5
    gt_ctry=gt_rois[:,1]+hs_gt*0.5
    
    bbox_deltas=np.zeros((rois.shape[0], 4), dtype=np.float32)
    bbox_deltas[:,0]=(gt_ctrx-ctrx)/ws
    bbox_deltas[:,1]=(gt_ctry-ctry)/hs
    
    bbox_deltas[:,2]=np.log(ws_gt/ws)
    bbox_deltas[:,3]=np.log(hs_gt/hs)
    
    return bbox_deltas

def bbox_transform_inv(boxes, bbox_deltas):
    if boxes.shape[0]==0:
        return np.zeros((0,4),dtype=bbox_deltas.dtype)
    
    ws=boxes[:,2]-boxes[:,0]+1
    hs=boxes[:,3]-boxes[:,1]+1
    
    ctrx=boxes[:,0]+0.5*ws
    ctry=boxes[:,1]+0.5*hs
    
    pred_boxes=np.zeros((boxes.shape[0], 4), dtype=bbox_deltas.dtype)
    
    pred_ctrx=bbox_deltas[:,0]*ws+ctrx
    pred_ctry=bbox_deltas[:,1]*hs+ctry
    
    pred_ws=np.exp(bbox_deltas[:,2])*ws
    pred_hs=np.exp(bbox_deltas[:,3])*hs
    
    pred_boxes[:,2]=pred_ctrx+pred_ws*0.5
    pred_boxes[:,3]=pred_ctry+pred_hs*0.5
    pred_boxes[:,0]=pred_ctrx-pred_ws*0.5
    pred_boxes[:,1]=pred_ctry-pred_hs*0.5
    
    return pred_boxes