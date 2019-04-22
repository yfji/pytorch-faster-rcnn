import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd.variable import Variable as Variable

import os
import numpy as np
import cv2

from model.faster_rcnn import FasterRCNN
from roidb import *
import rpn.generate_anchors as G
import rpn.anchor_target as T
import rpn.util as U
from core.config import cfg
from train import *

import logging

logging.basicConfig(level = logging.DEBUG,format = '%(asctime)s - %(levelname)s - %(message)s')

BBOX_OVERLAP_PRECOMPUTED=False
RESOLUTION={'VGG':(512,512),'RESNET':(512,512),'ALEX':(399,255)}
STRIDE={'VGG':8,'RESNET':8,'ALEX':8}
net_type='RESNET'

class TrainEngine(object):
    def __init__(self):
        self.batch_size=4
        cfg.PHASE='TRAIN'
        cfg[cfg.PHASE].IMS_PER_BATCH=self.batch_size
        self.stride=STRIDE[net_type]
        cfg.STRIDE=self.stride

        cfg.GAIN=0.02
        self.basic_size=cfg.BASIC_SIZE
        self.ratios=cfg.RATIOS
        self.scales=cfg.SCALES

        self.backbone_pretrained=True        
        self.lr_mult=0.5
        self.decay_ratio=0.1
        
        self.K=len(self.ratios)*len(self.scales)

        self.im_w, self.im_h=RESOLUTION[net_type]
        
        self.model=None

    def xy2wh(self, boxes):
        x=boxes[:,0]
        y=boxes[:,1]
        w=boxes[:,2]-x+1
        h=boxes[:,3]-y+1
        return x,y,w,h
            
    def get_param_groups(self, base_lr):
        backbone_lr=base_lr
        if self.backbone_pretrained:
            backbone_lr*=self.lr_mult
        return self.model.get_params({'backbone':backbone_lr, 'task':base_lr})
    
    def save_ckpt(self, stepvalues, epoch, iters, lr, logger):
        model_name='ckpt/model_{}.pkl'.format(int(iters))
        msg='Snapshotting to {}'.format(model_name)
        logger.info(msg)
        model={'model':self.model.state_dict(),'epoch':epoch, 'iters':iters, 'lr':lr, 'stepvalues':stepvalues, 'im_w':self.im_w, 'im_h':self.im_h}
        torch.save(model, model_name)
    
    def restore_from_ckpt(self, ckpt_path, logger):
        ckpt=torch.load(ckpt_path)
        epoch=ckpt['epoch']
        iters=ckpt['iters']
        lr=ckpt['lr']
        stepvalues=ckpt['stepvalues']
        msg='Restoring from {}\nStart epoch: {}, total iters: {}, current lr: {}'.format(ckpt_path,\
              epoch, iters, lr)
        logger.info(msg)
        logger.info('Model update successfully')
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
        coco_reader=COCODataReader(self.im_w, self.im_h, batch_size=self.batch_size)
        coco_loader=DataLoader(coco_reader, shuffle=True, batch_size=self.batch_size, num_workers=2)

        self.model=FasterRCNN(self.im_w, self.im_h)

        num_samples=coco_reader.__len__()
        epoch_iters=num_samples//self.batch_size

        num_epochs=50
        lr=0.0002
        stepvalues = [30]
        

        num_vis_anchors=100
        config_params={'lr':lr,'epoch':0,'start_epoch':0,'num_epochs':num_epochs,'epoch_iters':epoch_iters,\
            'out_size':coco_reader.out_size,'K':self.K,\
            'vis_anchors_dir':'./vis_anchors','display':20,
            'num_vis_anchors':num_vis_anchors}

        logger = logging.getLogger(__name__)
                
        logger.info('Load {} samples'.format(num_samples))
      
        if pretrained_model is None:
            self.model.init_weights()
        else:
            self.model.load_weights(model_path=pretrained_model)
        self.model.cuda()
        
        num_jumped=0
        
        iters_per_epoch=int(num_samples/self.batch_size)
        start_epoch=0
        
        if pretrained_model is not None:
            stepvalues, start_epoch, epoch_iters, lr=self.restore_from_ckpt(pretrained_model, logger)
        else:
            stepvalues, epoch_iters, lr=self.restore_from_epoch(start_epoch, stepvalues, lr, iters_per_epoch)
        
        if self.backbone_pretrained:    #backbone lr*=0.5
            params=self.get_param_groups(lr)
            optimizer=optim.SGD(params, lr=lr)
            for param_group in optimizer.param_groups:
                print('{} has learning rate {}'.format(param_group['key'], param_group['lr']))
        else:
            optimizer = optim.SGD(self.model.parameters(), lr=lr) 

        logger.info('Start training...')
        
        for epoch in range(start_epoch, num_epochs):
            config_params['epoch']=epoch
            train_epoch(self.model, coco_loader, optimizer, logger, config_params)
                
            if len(stepvalues)>0 and (epoch+1)==stepvalues[0]:
                lr*=self.decay_ratio
                config_params['lr']=lr
                msg='learning rate decay: %e'%lr
                logger.info(msg)
                for param_group in optimizer.param_groups:
                    param_group['lr']=lr
                    if self.backbone_pretrained and param_group['key']=='backbone':
                        param_group['lr']*=self.lr_mult
                stepvalues.pop(0)
            
        self.save_ckpt(stepvalues, epoch, epoch_iters, lr, logger)
        
        msg='Finish training!!\nTotal jumped batch: {}'.format(num_jumped)
        logger.info(msg)
            
if __name__=='__main__':
    pretrained='./ckpt/model_200000.pkl'
    engine=TrainEngine()
#    engine.train(pretrained_model=pretrained)
    engine.train()