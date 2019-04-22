import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.variable import Variable
import numpy as np
import cv2
from core.config import cfg
from model.faster_rcnn import FasterRCNN
import rpn.anchor_target as T
import rpn.util as U
from time import time

DEBUG=False

def draw_anchors(image, boxes, anchors, fg_anchor_inds, bg_anchor_inds):
    '''
    fg_anchors_inds: all inds in a frame, not a box
    '''
    n_instances=len(boxes)
    for i in range(n_instances):
        gt_box=boxes[i].astype(np.int32)
        cv2.rectangle(image, (gt_box[0],gt_box[1]),(gt_box[2],gt_box[3]), (0,0,255), 2)
    
    for fg_ind in fg_anchor_inds:
        anchor=anchors[fg_ind].astype(np.int32)
        ctrx=(anchor[0]+anchor[2])//2
        ctry=(anchor[1]+anchor[3])//2
        cv2.circle(image, (ctrx,ctry), 2, (255,0,0), -1)
        cv2.rectangle(image, (anchor[0],anchor[1]),(anchor[2],anchor[3]), (0,255,0), 1)

'''
params keys:
epoch
start_epoch
num_epochs
epoch_iters
lr
out_size
K
vis_anchors_dir
display
num_vis_anchors
'''

def train_epoch(model, data_loader, optimizer, logger, params):
    for iters, roidbs in enumerate(data_loader):
        B=len(roidbs)
        if B==0:
            msg='No labels in this minibatch, jump!'
            logger.info(msg)
        else:
            vis_image_list=[]
            bbox_overlaps=[]
            for db in roidbs:
                vis_image_list.append(db['data'].squeeze(0).astype(np.uint8, copy=True))
                if cfg.IMAGE_NORMALIZE:
                    db['data'] /= 255.0
                else:
                    db['data'] -= cfg.PIXEL_MEANS                    
                bbox_overlaps.append(db['bbox_overlaps'])
                
            output_dict=model(roidbs)                    
            gt_boxes=model.gt_boxes
            anchors=model.anchors
            
            tic=time()
            anchor_cls_targets, fg_anchor_inds, bg_anchor_inds, anchor_bbox_targets, bbox_weights=\
                T.compute_rpn_targets(anchors, gt_boxes, params['out_size'], K=params['K'], bbox_overlaps=bbox_overlaps, batch_size=B)
            
            anchor_cls_targets=Variable(torch.from_numpy(anchor_cls_targets).long().cuda())
            anchor_bbox_targets=Variable(torch.from_numpy(anchor_bbox_targets).float().cuda())                   
            toc=time()
            if DEBUG:
                print('Gen anchor targets cost {}s'.format(toc-tic))

            bbox_weights_var=Variable(torch.from_numpy(bbox_weights).cuda(), requires_grad=False)
            out_bbox=torch.mul(output_dict['rpn_bbox'], bbox_weights_var)
            anchor_bbox_targets=torch.mul(anchor_bbox_targets, bbox_weights_var)

            if params['epoch']==params['start_epoch'] and iters<params['num_vis_anchors']:
                anchors_per_image=np.split(anchors, B, axis=0)
                for i, canvas in enumerate(vis_image_list):
                    canvas_cpy=canvas.copy()
                    draw_anchors(canvas_cpy, gt_boxes[i], anchors_per_image[i], fg_anchor_inds[i], bg_anchor_inds[i])
                    cv2.imwrite('{}/vis_{}_{}.jpg'.format(params['vis_anchors_dir'], iters, i), canvas_cpy)
            
            '''
            In CrossEntropyLoss, input is BCHW, target is BHW, NOT BCHW!!! 
            '''
            num_examples=0
            for fg_inds in fg_anchor_inds:
                num_examples+=len(fg_inds)
            num_fg_proposals=output_dict['num_fgs']
            
            denominator_rpn=num_examples if num_examples>0 else 1
            denominator_frcnn=num_fg_proposals if num_fg_proposals>0 else 1
            denominator_rpn+=1e-4
            denominator_frcnn+=1e-4
            
            tic=time()
            rpn_loss_cls=F.cross_entropy(output_dict['rpn_logits'], anchor_cls_targets, size_average=True, ignore_index=-100)
            rpn_loss_bbox=F.smooth_l1_loss(out_bbox, anchor_bbox_targets, size_average=False, reduce=False)
            
            frcnn_loss_cls=F.cross_entropy(output_dict['frcnn_logits'], output_dict['frcnn_logits_target'])
            frcnn_loss_bbox=F.smooth_l1_loss(output_dict['frcnn_bbox'], output_dict['frcnn_bbox_target'], size_average=False, reduce=False)
            
            rpn_loss_bbox=torch.div(torch.sum(rpn_loss_bbox, dim=1), 4.0)
            rpn_loss_bbox=torch.div(torch.sum(rpn_loss_bbox), denominator_rpn)
            
            frcnn_loss_bbox=torch.div(torch.sum(frcnn_loss_bbox, dim=1), 4.0)
            frcnn_loss_bbox=torch.div(torch.sum(frcnn_loss_bbox), denominator_frcnn)  
            
            toc=time()
            if DEBUG:
                print('Compute loss costs {}s'.format(toc-tic))
            
            '''Do NOT multiply margin in RPN'''

            loss=rpn_loss_cls+rpn_loss_bbox+frcnn_loss_cls+frcnn_loss_bbox
            if iters%params['display']==0:
                msg='Epoch {}/{}. Iter_epoch {}/{}. Loss: {}. rpn_loss_cls: {}. rpn_loss_bbox: {}. frcnn_loss_cls: {}. frcnn_loss_bbox: {}. lr: {}. num_examples: {}. num_proposals: {}.'.\
                    format(params['epoch'], params['num_epochs'], iters, params['epoch_iters'],\
                                loss.item(), rpn_loss_cls.item(), rpn_loss_bbox.item(), \
                                frcnn_loss_cls.item(), frcnn_loss_bbox.item(), \
                                params['lr'], num_examples, num_fg_proposals)
                
                logger.info(msg)

            loss_val=loss.item()
                
            if loss_val > 1e5 or np.isnan(loss_val):
                msg='Loss too large or nan, stop! {}.'.format(loss_val)
                logger.error(msg)
                assert 0
            
            tic=time()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            toc=time()
            if DEBUG:
                print('Backward costs {}s'.format(toc-tic))
