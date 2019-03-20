import torch
import torch.nn as nn
#import torch.optim as optim
from torch.autograd.variable import Variable as Variable

import numpy as np
import cv2
#import matplotlib.pyplot as plt

from model.faster_rcnn import FasterRCNN
from model.faster_rcnn import nms_cuda
import rpn.generate_anchors as G
from rpn.util import bbox_transform_inv
from core.config import cfg

import logging
logging.basicConfig(level = logging.DEBUG,format = '%(asctime)s - %(levelname)s - %(message)s')

#CLASSES=['__background', 'car', 'van', 'bus', 'truck']
CLASSES=CLASSES=['__background__','bike', 
                        'bus', 
                        'car',
                        'motor', 
                        'person', 
                        'rider', 
                        'traffic light', 
                        'traffic sign', 
                        'train', 
                        'truck']
COLORS=[None, (0,0,255), (0,255,0), (255,0,0), (0,255,255)]

def im_detect(image, anchors, model):
    im_w=model.im_width
    im_h=model.im_height

    h,w=image.shape[:2]
    image=cv2.resize(image, (im_w,im_h), interpolation=cv2.INTER_LINEAR)
    
    xscale=1.0*im_w/w
    yscale=1.0*im_h/h
    
    roidb={}
    roidb['image']=image[np.newaxis,:,:,:].astype(np.float32)
    if cfg.IMAGE_NORMALIZE:
        roidb['image'] -= roidb['image'].min()
        roidb['image'] /= roidb['image'].max()
        roidb['image']=(roidb['image']-0.5)/0.5
    else:
        roidb['image'] -= cfg.PIXEL_MEANS

    roidb['gt_boxes']=None
    roidb['gt_classes']=None
    roidb['bbox_overlaps']=None
    roidb['anchors']=anchors

    output_dict=model([roidb])

    frcnn_probs=output_dict['frcnn_probs'].cpu().data.numpy()
    frcnn_bbox=output_dict['frcnn_bbox'].cpu().data.numpy()

    proposals= output_dict['proposals']
    num_proposals_per_image=output_dict['num_proposals']
    
    print(num_proposals_per_image)
    print(frcnn_probs.shape)
    
    cls_bboxes=[[]]
#    classes=np.argmax(frcnn_probs[:,1:], axis=1)+1
#    max_probs=np.max(frcnn_probs[:,1:], axis=1)
    classes=np.argmax(frcnn_probs, axis=1)
    max_probs=np.max(frcnn_probs, axis=1)
    
#    print(classes)

#    bbox_pred=bbox_transform_inv(proposals, frcnn_bbox)

    for i in range(1, cfg.NUM_CLASSES):
        cls_inds=np.where(classes==i)[0]
        if cls_inds.size==0:
            cls_bboxes.append([])
        else:
            cls_proposals=proposals[cls_inds]
            cls_frcnn_bbox=frcnn_bbox[cls_inds,4*i:4*i+4]
            
            if cfg.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                cls_frcnn_bbox = cls_frcnn_bbox*cfg.BBOX_STD_DEV
                
            cls_bbox_pred=bbox_transform_inv(cls_proposals, cls_frcnn_bbox)
            cls_probs=max_probs[cls_inds]
            
#            print(frcnn_probs[cls_inds])
    
            cls_bbox_pred=np.hstack((cls_bbox_pred, cls_probs.reshape(-1,1)))
            
            order=np.argsort(cls_probs)[::-1]
            bbox_order=cls_bbox_pred[order]
            
            pick=nms_cuda(bbox_order, nms_thresh=cfg.TEST.NMS_THRESH, xyxy=True)
#            pick=pick[:600]
            
            bboxes=bbox_order[pick].reshape(-1,bbox_order.shape[1])
    
            print('Class {} has {} instances!'.format(CLASSES[i], bboxes.shape[0]))
            cls_bboxes.append(bboxes)

    print(len(proposals))
    return cls_bboxes, (xscale, yscale), proposals

def draw_boxes(image, cls_bboxes, xyscale):
    for i in range(1, cfg.NUM_CLASSES):
        bbox_with_score=cls_bboxes[i]
        if len(bbox_with_score)==0:
            continue
#        print(bbox_with_score)
        scores=bbox_with_score[:,-1]
        fg_inds=np.where(scores>0.3)
        bbox_with_score=bbox_with_score[fg_inds]
        x1,y1,x2,y2,scores=np.split(bbox_with_score, 5, axis=1)

        x1=(x1/xyscale[0]).astype(np.int32)
        x2=(x2/xyscale[0]).astype(np.int32)
        y1=(y1/xyscale[1]).astype(np.int32)
        y2=(y2/xyscale[1]).astype(np.int32)

        for j in range(bbox_with_score.shape[0]):
            cv2.rectangle(image, (x1[j],y1[j]),(x2[j],y2[j]), COLORS[i%len(COLORS)], 3)
            cv2.putText(image, '{}:{}'.format(CLASSES[i], scores[j]), (x1[j], y1[j]-10), cv2.FONT_HERSHEY_PLAIN, 1.0, COLORS[i%len(COLORS)], 1)

    
def draw_proposals(image, proposals, xyscale):
    boxes=proposals[:,:4]
    boxes[:,[0,2]]/=xyscale[0]
    boxes[:,[1,3]]/=xyscale[1]
    for box in boxes:
        box=box.astype(np.int32)
        cv2.rectangle(image, (box[0],box[1]),(box[2],box[3]), (0,255,0), 1)
        
if __name__=='__main__':
    im_width=640
    im_height=384
    stride=8
    basic_size = cfg.BASIC_SIZE
    ratios=cfg.RATIOS
    scales=cfg.SCALES
    stride = 8
    
    cfg.STRIDE=stride
    cfg.PHASE='TEST'
    cfg.TEST.RPN_POST_NMS_TOP_N=300
    cfg.NUM_CLASSES=len(CLASSES)
    cfg.TEST.IMS_PER_BATCH=1
    cfg.TEST.NMS_THRESH=0.5
    cfg.TEST.RPN_NMS_THRESH=0.7
#    cfg.TEST.RPN_POST_NMS_TOP_N=300

    K=len(ratios)*len(scales)

    raw_anchors=G.generate_anchors(basic_size, ratios, scales)

    bound=(im_width, im_height)
    out_size=(im_width//stride, im_height//stride)

    dummy_search_box=np.array([0,0,bound[0],bound[1]]).reshape(1,-1)
    anchors=G.gen_region_anchors(raw_anchors, dummy_search_box, bound, K=K, size=out_size)[0]
    
    print(anchors.shape)

#    img_files=['img00337.jpg','img00832.jpg','img00995.jpg','img01879.jpg','road.jpg']
    img_files=['road.jpg']
    model_path='./ckpt/model_660000.pkl'
    model=FasterRCNN(im_width, im_height, pretrained=False)
    model.load_weights(model_path=model_path)
    model.cuda()

    for f in img_files:
      img_path='./images/{}'.format(f)
      image=cv2.imread(img_path)
      canvas=image.copy()
      
      cls_bboxes, xyscale, proposals=im_detect(image, anchors, model)
  
#      draw_boxes(canvas, cls_bboxes, xyscale)
      draw_proposals(canvas, proposals, xyscale)
#      cv2.imwrite('detect.jpg', canvas)
      cv2.imwrite('proposals.jpg',canvas)
      cv2.waitKey(0)

