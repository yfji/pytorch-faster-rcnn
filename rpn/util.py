import numpy as np

DEBUG=True

def bbox_overlap_rpn(anchors, boxes):
    n_anchor=len(anchors)
    n_boxes=len(boxes)
    bbox_overlaps=np.zeros((n_anchor, n_boxes), dtype=np.float32)
    
    anchors_per_box=np.split(anchors, n_boxes, axis=0)
    starts=np.split(np.arange(len(anchors)),n_boxes, axis=0)

    for i, box in enumerate(boxes):
        anchors_this_box=anchors_per_box[i]
        anchor_xmins=anchors_this_box[:,0]
        anchor_ymins=anchors_this_box[:,1]
        anchor_ws=anchors_this_box[:,2]-anchor_xmins+1
        anchor_hs=anchors_this_box[:,3]-anchor_ymins+1
  
        anchor_areas=anchor_ws*anchor_hs
        if not np.any(box):
            bbox_overlaps[:,i]=0.0
        else:
            xmin=np.maximum(anchor_xmins,box[0])
            ymin=np.maximum(anchor_ymins,box[1])
            xmax=np.minimum(anchors_this_box[:,2],box[2])
            ymax=np.minimum(anchors_this_box[:,3],box[3])
            
            ws=np.maximum(0,xmax-xmin+1)
            hs=np.maximum(0,ymax-ymin+1)
            intersec=ws*hs
            
            box_area=(box[2]-box[0]+1)*(box[3]-box[1]+1)
            union_areas=anchor_areas+box_area-intersec
            overlap=1.0*intersec/union_areas
            bbox_overlaps[starts[i][0]:starts[i][-1]+1,i]=overlap[...]
    
    return bbox_overlaps

def bbox_overlap_frcnn(anchors, boxes):
    n_anchor=len(anchors)
    n_boxes=len(boxes)
    
#    print(anchors[:10])
    bbox_overlaps=np.zeros((n_anchor, n_boxes), dtype=np.float32)
    anchor_xmins=anchors[:,0]
    anchor_ymins=anchors[:,1]
    anchor_ws=anchors[:,2]-anchor_xmins+1
    anchor_hs=anchors[:,3]-anchor_ymins+1

    anchor_areas=anchor_ws*anchor_hs
    
    if DEBUG:
        invalid_inds=np.where(np.bitwise_or(anchor_ws<=0, anchor_hs<=0)==1)[0]
        if len(invalid_inds)>0:
            print('Invalid anchors')
            print(anchors[invalid_inds])
            
    for i, box in enumerate(boxes):
        if not np.any(box):
            bbox_overlaps[:,i]=0.0
        else:
            xmin=np.maximum(anchor_xmins,box[0])
            ymin=np.maximum(anchor_ymins,box[1])
            xmax=np.minimum(anchors[:,2],box[2])
            ymax=np.minimum(anchors[:,3],box[3])
            
            ws=np.maximum(0,xmax-xmin+1)
            hs=np.maximum(0,ymax-ymin+1)
            intersec=ws*hs
            
            box_area=(box[2]-box[0]+1)*(box[3]-box[1]+1)
            union_areas=anchor_areas+box_area-intersec
    #        anchor_areas=np.minimum(box_area, anchor_ws*anchor_hs)
    #        anchor_areas=anchor_ws*anchor_hs
            overlap=1.0*intersec/union_areas
            
            bbox_overlaps[:,i]=overlap[...]
        
    return bbox_overlaps

def bbox_overlaps_per_image(anchors, boxes, branch='rpn'):
    if branch=='rpn':
        return bbox_overlap_rpn(anchors, boxes)
    elif branch=='frcnn':
        return bbox_overlap_frcnn(anchors, boxes)
 
def bbox_overlaps_batch(anchors, boxes, batch_size, branch='rpn'):
    bbox_overlaps_batch=[]
    anchors_per_image=np.split(anchors, batch_size, axis=0)
    for i in range(batch_size):
        bbox_overlaps=bbox_overlaps_per_image(anchors_per_image[i], boxes[i], branch) 
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