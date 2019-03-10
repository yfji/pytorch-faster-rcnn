import numpy as np
from rpn.util import bbox_transform_inv

penalty_k = 0.055
window_influence = 0.4
'''
one box and anchors
'''
def gen_proposals(boxes, bbox_deltas, bound, min_size=8):
#    proposals=bbox_transform_inv(boxes,bbox_deltas)
#    proposals[:,0]=np.maximum(0, proposals[:,0])
#    proposals[:,1]=np.maximum(0, proposals[:,1])
#    proposals[:,2]=np.minimum(bound[0], proposals[:,2])
#    proposals[:,3]=np.minimum(bound[1], proposals[:,3])
#    
#    return proposals
    proposals=bbox_transform_inv(boxes, bbox_deltas)
    proposals=np.hstack((proposals, boxes[:,-1].reshape(-1,1)))
    
    proposals[:,0]=np.maximum(0, proposals[:,0])
    proposals[:,1]=np.maximum(0, proposals[:,1])
    proposals[:,2]=np.minimum(bound[0], proposals[:,2])
    proposals[:,3]=np.minimum(bound[1], proposals[:,3])
    
    ws=proposals[:,2]-proposals[:,0]
    hs=proposals[:,3]-proposals[:,1]
    
    keep=np.where(np.bitwise_and(ws>=min_size, hs>=min_size)==1)[0]
    proposals_keep=proposals[keep]
    boxes_keep=boxes[keep]
    
#    print(proposals[:30])
    assert len(keep)>0, 'Anchors bbox_transform_inv have all rois of negative sides'
    
    return proposals_keep, boxes_keep

'''
select correct proposals according to
1.Distance
2.Cosine window
'''
def select_proposals(ref_box, proposals, dist_thresh=20, top_n=50):
    '''already sorted'''
    w_box=ref_box[2]-ref_box[0]+1
    h_box=ref_box[3]-ref_box[1]+1
    
    ws=proposals[:,2]-proposals[:,0]+1
    hs=proposals[:,3]-proposals[:,1]+1
    
    ref_ctrx=ref_box[0]+w_box*0.5
    ref_ctry=ref_box[1]+h_box*0.5
    ctrx=proposals[:,0]+ws*0.5
    ctry=proposals[:,1]+hs*0.5
    
    dist_x=ctrx-ref_ctrx
    dist_y=ctry-ref_ctry
    
    keep=np.bitwise_and(np.abs(dist_x)<dist_thresh, np.abs(dist_y)<dist_thresh)
    proposals_keep=proposals[keep]
    
#    scores=proposals_keep[:,-1]
    ratio_proposals=1.0*hs[keep]/ws[keep]
    scale_proposals=np.sqrt(ws[keep]*hs[keep])
    ratio_box=1.0*h_box/w_box
    scale_box=np.sqrt(w_box*h_box)
    
    penalty=np.maximum(ratio_proposals/ratio_box, ratio_box/ratio_proposals)*np.maximum(scale_proposals/scale_box,scale_box/scale_proposals)
#    pscores=scores*(1.0/penalty)
    order_penalty=np.argsort(penalty)
    proposals_keep=proposals_keep[order_penalty[:top_n]]
#    best_proposal=np.mean(proposals_keep, axis=0)
    return proposals_keep
#    return best_proposal[np.newaxis,:]

'''
https://github.com/songdejia/Siamese-RPN-pytorch/blob/master/code/run_SiamRPN.py
'''
def select_proposals_cosine(ref_box, proposals, rpn_conv_size, K):
    def change(r):
        return np.maximum(r, 1./r)
    def sz(w, h):
#        pad = (w + h) * 0.5
#        sz2 = (w + pad) * (h + pad)
        return w*h

    window=np.outer(np.hanning(rpn_conv_size), np.hanning(rpn_conv_size))
    window=np.tile(window.flatten(), K)
    scores=proposals[:,-1]

    w=ref_box[2]-ref_box[0]+1
    h=ref_box[3]-ref_box[1]+1

    ws=proposals[:,2]-proposals[:,0]+1
    hs=proposals[:,3]-proposals[:,1]+1

    s_c=change(sz(ws,hs)/sz(w,h))
    r_c=change((1.0*ws/hs)/(1.0*w/h))

    penalty = np.exp(-(r_c * s_c - 1.) * penalty_k)
    pscore=penalty*scores
    pscore = pscore * (1 - window_influence) + window * window_influence
    best_pscore_id = np.argmax(pscore)

    print('Best score: {}'.format(scores[best_pscore_id]))
    return proposals[best_pscore_id][np.newaxis,:]

def select_proposals_overlap(ref_box, proposals):
    w_box=ref_box[2]-ref_box[0]+1
    h_box=ref_box[3]-ref_box[1]+1
    
    box_area=w_box*h_box

    ws=proposals[:,2]-proposals[:,0]+1
    hs=proposals[:,3]-proposals[:,1]+1

#    scores=proposals[:,-1]
    areas=ws*hs

    xmax=np.minimum(ref_box[2], proposals[:,2])
    ymax=np.minimum(ref_box[3], proposals[:,3])
    xmin=np.maximum(ref_box[0], proposals[:,0])
    ymin=np.maximum(ref_box[1], proposals[:,1])

    w=np.maximum(0, xmax-xmin)
    h=np.maximum(0, ymax-ymin)

    intersecs=w*h

    ious=intersecs/(areas+box_area-intersecs)
    
    keep=np.where(ious>0.7)[0]
    proposals=proposals[keep]
#    scores=scores[keep]
#    ious=ious[keep]
#    
#    iscores=scores*(1-window_influence)+window_influence*ious
#    best_score_id=np.argmax(iscores)
#    print('Best score: {}. Best iscore: {}'.format(scores[best_score_id], iscores[best_score_id]))
#    return proposals[best_score_id][np.newaxis,:]
    best_proposal=np.mean(proposals, axis=0)
#    return proposals
    return best_proposal[np.newaxis,:]
