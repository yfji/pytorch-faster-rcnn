import numpy as np
from rpn.util import bbox_transform
from rpn.util import bbox_overlaps_batch
from core.config import cfg
'''
anchors_with_shift
[anchors_box1, anchors_box2,...,anchors_box_n]
aranged by the search_boxes of last frame

gt_boxes
[gt_box1, gt_box2,...,gt_box_n]
aranged by the gt_boxes of current frame
'''

def compute_rpn_targets(anchors, gt_boxes, out_size, K=9, bbox_overlaps=None, batch_size=1):
    anchor_cls_targets=np.zeros((0, K*out_size[1], out_size[0]), dtype=np.int32)
    anchor_bbox_targets=np.zeros((0, 4*K, out_size[1], out_size[0]),dtype=np.float32)
    fg_anchor_inds=[]
    bg_anchor_inds=[]

    if bbox_overlaps is None:
        bbox_overlaps=bbox_overlaps_batch(anchors, gt_boxes, batch_size, branch='frcnn')
        
    bbox_weights=np.zeros((0,4*K, out_size[1], out_size[0]), dtype=np.float32)
    
    anchors_per_image=np.split(anchors, batch_size, axis=0)
    for i in range(batch_size):
        anchors_this_image=anchors_per_image[i]  #all anchors per ref box!!!!!
        bbox_overlap=np.asarray(bbox_overlaps[i])
        max_overlap=np.max(bbox_overlap, axis=1)

        '''cls start'''
        anchor_cls_target=np.zeros(anchors_this_image.shape[0], dtype=np.int32)-100    #box_size*A*K-->N*A*K
#        print(max_overlap)
        fg_anchor_ind=np.where(max_overlap>=cfg[cfg.PHASE].RPN_POSITIVE_THRESH)[0]
        bg_anchor_ind=np.where(max_overlap<cfg[cfg.PHASE].RPN_NEGATIVE_THRESH)[0]

        if cfg.CHOOSE_ANCHOR:
            fg_rois_this_image=min(fg_anchor_ind.size, int(cfg.POS_ANCHOR_FRACTION*cfg.ANCHOR_NUM))
            bg_rois_this_image=min(bg_anchor_ind.size, cfg.ANCHOR_NUM-fg_rois_this_image)
            if fg_rois_this_image>0:
                fg_anchor_ind=np.random.choice(fg_anchor_ind, size=fg_rois_this_image, replace=False)
            bg_anchor_ind=np.random.choice(bg_anchor_ind, size=bg_rois_this_image, replace=False)

        fg_anchor_inds.append(fg_anchor_ind)
        bg_anchor_inds.append(bg_anchor_ind)

        anchor_cls_target[fg_anchor_ind]=1
        anchor_cls_target[bg_anchor_ind]=0
        
        anchor_cls_target=anchor_cls_target.reshape(1, out_size[1], out_size[0], K).\
            transpose(0,3,2,1).reshape(1, K*out_size[1], out_size[0])
        anchor_cls_targets=np.append(anchor_cls_targets, anchor_cls_target, 0)
        '''cls end'''
        '''bbox start'''
#        bbox_loss_inds=np.where(max_overlap>=cfg[cfg.PHASE].RPN_BBOX_THRESH)[0]
        bbox_loss_inds=fg_anchor_ind
        mask_inds=np.zeros((anchors_this_image.shape[0], 4), dtype=np.float32)
        mask_inds[bbox_loss_inds,:]=1
        gt_boxes_sample=gt_boxes[i]
        
        gt_rois=gt_boxes_sample[np.argmax(bbox_overlap, axis=1)]
        bbox_deltas=bbox_transform(anchors_this_image, gt_rois)
        if cfg.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            bbox_deltas=bbox_deltas/cfg.RPN_BBOX_STD_DEV
            
#        bbox_deltas*=mask_inds  
        bbox_deltas=bbox_deltas.reshape(1, out_size[1], out_size[0], 4*K).transpose((0,3,1,2))
        anchor_bbox_targets=np.append(anchor_bbox_targets, bbox_deltas, 0)
        '''bbox end'''
        '''bbox weights'''
        mask_inds=mask_inds.reshape(1, out_size[1], out_size[0], 4*K).transpose(0,3,1,2)
        bbox_weights=np.append(bbox_weights, mask_inds, 0)
    return anchor_cls_targets, fg_anchor_inds, bg_anchor_inds, anchor_bbox_targets, bbox_weights