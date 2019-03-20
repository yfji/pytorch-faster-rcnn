import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd.variable import Variable as Variable

import os
import numpy as np
import cv2

from model.faster_rcnn import FasterRCNN, VGG_PRETRAINED, VGG_PRETRAINED_BN
from roidb.data_loader import DataLoader
from roidb.detrac_data_loader import DetracDataLoader
from roidb.deepdrive_data_loader import DeepDriveDataLoader
import rpn.generate_anchors as G
import rpn.anchor_target as T
import rpn.util as U
from core.config import cfg

import logging

logging.basicConfig(level = logging.DEBUG,format = '%(asctime)s - %(levelname)s - %(message)s')

BBOX_OVERLAP_PRECOMPUTED=False
RESOLUTION={'VGG':(640,384),'RESNET':(640,384),'ALEX':(399,255)}
STRIDE={'VGG':8,'RESNET':8,'ALEX':8}
net_type='RESNET'

class TrainEngine(object):
    def __init__(self):
        self.batch_size=2
        cfg.PHASE='TRAIN'
        cfg[cfg.PHASE].IMS_PER_BATCH=self.batch_size
        self.stride=STRIDE[net_type]
        cfg.STRIDE=self.stride

        cfg.GAIN=0.02
        self.basic_size=cfg.BASIC_SIZE
        self.ratios=cfg.RATIOS
        self.scales=cfg.SCALES
        self.display=20
        self.snapshot=20000
        self.decay_ratio=0.1
        
        self.lr_mult=0.5
        
        self.K=len(self.ratios)*len(self.scales)

        self.im_w, self.im_h=RESOLUTION[net_type]
        
        self.raw_anchors=G.generate_anchors(self.basic_size, self.ratios, self.scales)
        
        self.model=FasterRCNN(self.im_w, self.im_h)

    def xy2wh(self, boxes):
        x=boxes[:,0]
        y=boxes[:,1]
        w=boxes[:,2]-x+1
        h=boxes[:,3]-y+1
        return x,y,w,h

    def gen_anchors(self, search_boxes, bound):
        xs,ys,ws,hs=self.xy2wh(search_boxes)
        
        box_anchors=[]
        roi_size = self.model.rpn_conv_size

        for i in range(len(search_boxes)):
            A=roi_size**2
            K=self.K
            shifts_ctrs = G.calc_roi_align_shifts(search_boxes[i], roi_size, bound, stride=self.stride)
            anchors = self.raw_anchors.reshape((1, K, 4)) + shifts_ctrs.reshape((1, A, 4)).transpose((1, 0, 2))
            anchors = anchors.reshape((A*K, 4))
            box_anchors.append(anchors)
        
        return box_anchors
     
    def vis_anchors(self, image, boxes, anchors, fg_anchor_inds, bg_anchor_inds):
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
            
    def get_param_groups(self, base_lr):
        backbone_lr=base_lr
        if os.path.exists(VGG_PRETRAINED_BN):
            backbone_lr*=self.lr_mult
        return self.model.get_params({'backbone':backbone_lr, 'task':base_lr})
    
    def save_ckpt(self, stepvalues, epoch, iters, lr, logger, log_file):
#        model_name='ckpt_{}/dl_mot_iter_{}.pkl'.format(dataset, int(iters))
        model_name='ckpt/model_{}.pkl'.format(int(iters))
        msg='Snapshotting to {}'.format(model_name)
        logger.info(msg)
        log_file.write(msg+'\n')
        model={'model':self.model.state_dict(),'epoch':epoch, 'iters':iters, 'lr':lr, 'stepvalues':stepvalues, 'im_w':self.im_w, 'im_h':self.im_h}
        torch.save(model, model_name)
    
    def restore_from_ckpt(self, ckpt_path, logger, log_file):
        ckpt=torch.load(ckpt_path)
#        model=ckpt['model']
        epoch=ckpt['epoch']
        iters=ckpt['iters']
        lr=ckpt['lr']
        stepvalues=ckpt['stepvalues']
        msg='Restoring from {}\nStart epoch: {}, total iters: {}, current lr: {}'.format(ckpt_path,\
              epoch, iters, lr)
        logger.info(msg)
        log_file.write(msg+'\n')
#        self.model.update_state_dict(model)
        msg='Model update successfully'
        logger.info(msg)
        log_file.write(msg+'\n')
        return stepvalues, epoch, iters, lr
    
    def restore_from_epoch(self, start_epoch, stepvalues, lr, iters_per_epoch):
        epoch_iters=int(start_epoch*iters_per_epoch)
        base_lr=lr
        stepvalues_new=[]
        for i in stepvalues:
            if i>start_epoch:
                stepvalues_new.append(i)
            else:
                lr*=self.decay_ratio
        print('lr drops from {} to {}'.format(base_lr, lr))
        print('Restoring from epoch {}, iters {}'.format(start_epoch, epoch_iters))
        stepvalues=stepvalues_new
        return stepvalues, epoch_iters, lr
        
    def train(self, pretrained_model=None):
        dataset='deepdrive'
        if dataset=='detrac':
            loader=DetracDataLoader(self.im_w, self.im_h, batch_size=self.batch_size)
        elif dataset=='deepdrive':
            loader=DeepDriveDataLoader(self.im_w, self.im_h, batch_size=self.batch_size)

        logger = logging.getLogger(__name__)

        backbone_pretrained=True
        
        num_samples=loader.get_num_samples()
        
        logger.info('Load {} samples'.format(num_samples))
        
        num_epochs=50
        lr=0.0002
        stepvalues = [30]

        if pretrained_model is None:
            self.model.init_weights()
        else:
            self.model.load_weights(model_path=pretrained_model)
        self.model.cuda()
        
        num_vis_anchors=100
        
        num_jumped=0
        log_file=open('loss.log','w')
        
        iters_per_epoch=int(num_samples/self.batch_size)
        start_epoch=0
        
        if pretrained_model is not None:
            stepvalues, start_epoch, epoch_iters, lr=self.restore_from_ckpt(pretrained_model, logger, log_file)
        else:
            stepvalues, epoch_iters, lr=self.restore_from_epoch(start_epoch, stepvalues, lr, iters_per_epoch)
        
        if backbone_pretrained:
            params=self.get_param_groups(lr)
            optimizer=optim.SGD(params, lr=lr)
            for param_group in optimizer.param_groups:
                print('{} has learning rate {}'.format(param_group['key'], param_group['lr']))
        else:
            optimizer = optim.SGD(self.model.parameters(), lr=lr) 
        logger.info('Start training...')
        
        for epoch in range(start_epoch, num_epochs):
            data_iterator=DataLoader(loader)
            for iters, roidbs in enumerate(data_iterator):
                B=len(roidbs)
                if B==0:
                    msg='No reference image in this minibatch, jump!'
#                    logger.info(msg)
                    log_file.write(msg+'\n')
                    num_jumped+=1
                else:
                    '''NHWC'''
                    vis_image_list=[]
                    bbox_overlaps=[]
                    for db in roidbs:
                        vis_image_list.append(db['image'].squeeze(0).astype(np.uint8, copy=True))
                        if cfg.IMAGE_NORMALIZE:
                            db['image'] -= db['image'].min()
                            db['image'] /= db['image'].max()
                            db['image']=(db['image']-0.5)/0.5
                        else:
                            db['image'] -= cfg.PIXEL_MEANS
                            
                        bbox_overlaps.append(db['bbox_overlaps'])
                        
#                    out_cls, out_bbox, proposals=self.model(roidbs)                    
                    output_dict=self.model(roidbs)                    
                    gt_boxes=self.model.gt_boxes
#                    gt_classes=self.model.gt_classes
                    out_size=self.model.out_size
#                    num_boxes=self.model.num_boxes

                    anchors=self.model.anchors
                    
                    anchor_cls_targets, fg_anchor_inds, bg_anchor_inds, anchor_bbox_targets, bbox_weights=\
                        T.compute_rpn_targets(anchors, gt_boxes, out_size, K=self.K, bbox_overlaps=bbox_overlaps, batch_size=self.batch_size)
                    
                    anchor_cls_targets=Variable(torch.from_numpy(anchor_cls_targets).long().cuda())
                    anchor_bbox_targets=Variable(torch.from_numpy(anchor_bbox_targets).float().cuda())                   

                    bbox_weights_var=Variable(torch.from_numpy(bbox_weights).cuda(), requires_grad=False)
                    out_bbox=torch.mul(output_dict['rpn_bbox'], bbox_weights_var)
                    anchor_bbox_targets=torch.mul(anchor_bbox_targets, bbox_weights_var)

                    if epoch==start_epoch and iters<num_vis_anchors:
                        anchors_per_image=np.split(anchors, self.batch_size, axis=0)
                        for i, canvas in enumerate(vis_image_list):
                            canvas_cpy=canvas.copy()
                            self.vis_anchors(canvas_cpy, gt_boxes[i], anchors_per_image[i], fg_anchor_inds[i], bg_anchor_inds[i])
                            cv2.imwrite('vis_anchors/vis_{}_{}.jpg'.format(iters, i), canvas_cpy)
                    
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

                    rpn_loss_cls=F.cross_entropy(output_dict['rpn_logits'], anchor_cls_targets, size_average=True, ignore_index=-100)
                    rpn_loss_bbox=F.smooth_l1_loss(out_bbox, anchor_bbox_targets, size_average=False, reduce=False)
                    
                    frcnn_loss_cls=F.cross_entropy(output_dict['frcnn_logits'], output_dict['frcnn_logits_target'])
                    frcnn_loss_bbox=F.smooth_l1_loss(output_dict['frcnn_bbox'], output_dict['frcnn_bbox_target'], size_average=False, reduce=False)
                    
                    rpn_loss_bbox=torch.div(torch.sum(rpn_loss_bbox, dim=1), 4.0)
                    rpn_loss_bbox=torch.div(torch.sum(rpn_loss_bbox), denominator_rpn)
                    
                    frcnn_loss_bbox=torch.div(torch.sum(frcnn_loss_bbox, dim=1), 4.0)
                    frcnn_loss_bbox=torch.div(torch.sum(frcnn_loss_bbox), denominator_frcnn)  
                    
                    '''Do NOT multiply margin in RPN'''
    
                    loss=rpn_loss_cls+rpn_loss_bbox+frcnn_loss_cls+frcnn_loss_bbox
                    if iters%self.display==0:
                        msg='Epoch {}/{}. Iter_epoch {}/{}. Global_iter: {}. Loss: {}. rpn_loss_cls: {}. rpn_loss_bbox: {}. frcnn_loss_cls: {}. frcnn_loss_bbox: {}. lr: {}. num_examples: {}. num_proposals: {}. Batch size: {}/{}. FG_THRESH: {}'.format(epoch, num_epochs, iters, iters_per_epoch, epoch_iters, \
                                       loss.item(), rpn_loss_cls.item(), rpn_loss_bbox.item(), \
                                       frcnn_loss_cls.item(), frcnn_loss_bbox.item(), \
                                        lr, num_examples, num_fg_proposals, B, self.batch_size, cfg[cfg.PHASE].FG_THRESH)
                        
                        logger.info(msg)
                        log_file.write(msg+'\n')

                    loss_val=loss.item()
                        
                    if loss_val > 1e7:
                        msg='Loss too large, stop! {}.'.format(loss_val)
                        logger.error(msg)
                        log_file.write(msg+'\n')
                        assert 0
                    if np.isnan(loss_val):
                        msg='Loss nan, stop! {}.'.format(loss_val)
                        logger.error(msg)
                        log_file.write(msg+'\n')
                        assert 0
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                epoch_iters+=1
                
                if epoch_iters%self.snapshot==0:
                    self.save_ckpt(stepvalues, epoch, epoch_iters, lr, logger, log_file)
                
            if len(stepvalues)>0 and (epoch+1)==stepvalues[0]:
                lr*=self.decay_ratio
                msg='learning rate decay: %e'%lr
                logger.info(msg)
                log_file.write(msg+'\n')
                for param_group in optimizer.param_groups:
                    param_group['lr']=lr
                    if backbone_pretrained and param_group['key']=='backbone':
                        param_group['lr']*=self.lr_mult
                stepvalues.pop(0)
            
        self.save_ckpt(stepvalues, epoch, epoch_iters, lr, logger, log_file)
        
        msg='Finish training!!\nTotal jumped batch: {}'.format(num_jumped)
        logger.info(msg)
        log_file.write(msg+'\n')
        log_file.close()
            
if __name__=='__main__':
    pretrained='./ckpt/model_200000.pkl'
    engine=TrainEngine()
#    engine.train(pretrained_model=pretrained)
    engine.train()