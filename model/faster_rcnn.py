'''
Siamese-RPN

@author: yfji

2018.9.1.21
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.variable import Variable
from time import time

import sys
sys.path.insert(0,'/home/yfji/Workspace/PyTorch/Faster-RCNN')

import math
import numpy as np
from collections import OrderedDict
from model.vgg16 import Vgg16
import model.resnet as resnet
import model.fpn as fpn
from model.my_fast_rcnn import FastRCNN
from rpn.util import bbox_transform_inv
#from rpn.nms import nms
from fast_rcnn.proposal_target import get_proposal_target
from core.config import cfg

from nms.nms_wrapper import nms
from roialign.roi_align.crop_and_resize import CropAndResizeFunction

DEBUG=False

def crop_and_resize(pool_size, feature_map, boxes, box_ind):
    if boxes.shape[1]==5:
        x1, y1, x2, y2, _= boxes.chunk(5, dim=1)
    else:
        x1, y1, x2, y2= boxes.chunk(4, dim=1)
    im_h, im_w=feature_map.shape[2:4]
    x1=x1/(float(im_w-1))
    x2=x2/(float(im_w-1))
    y1=y1/(float(im_h-1))
    y2=y2/(float(im_h-1))

    boxes = torch.cat((y1, x1, y2, x2), 1)
    return CropAndResizeFunction(pool_size[0],pool_size[1],0)(feature_map, boxes, box_ind)

def nms_cuda(boxes_np, nms_thresh=0.7, xyxy=True):    
    if xyxy:
        x1,y1,x2,y2,scores=np.split(boxes_np, 5, axis=1)
        boxes_np=np.hstack([y1,x1,y2,x2,scores])
    boxes_pth=torch.from_numpy(boxes_np).float().cuda()
    pick=nms(boxes_pth, nms_thresh)
    pick=pick.cpu().data.numpy()
    if len(pick.shape)==2:
        pick=pick.squeeze()
    return pick

class FasterRCNN(nn.Module):
    def __init__(self, im_width, im_height, pretrained=True):
        super(FasterRCNN, self).__init__()
        self.frcnn_roi_size=cfg.FRCNN_ROI_SIZE
        self.batch_size=cfg[cfg.PHASE].IMS_PER_BATCH

        self.rpn_out_ch=512
        self.features_out_ch=256

        self.im_width=im_width
        self.im_height=im_height
        self.bound=(im_width, im_height)
        self.stride=cfg.STRIDE
        self.K=len(cfg.RATIOS)*len(cfg.SCALES)
        self.use_bn=not cfg.IMAGE_NORMALIZE
        
        self.out_width=self.im_width//self.stride
        self.out_height=self.im_height//self.stride
        self.out_size=(self.out_width, self.out_height)

        self.num_anchors=self.K*(self.out_width*self.out_height)

        self.fpn=fpn.FPN(self.features_out_ch)
        self.features=resnet.resnet50(pretrained=pretrained)
        self.make_net()
        
    def load_weights(self, model_path=None):
        print('loading model from {}'.format(model_path))
        pretrained_dict = torch.load(model_path)
        keys=list(pretrained_dict.keys())
        if 'epoch' in keys:
            print('Restoring model from self-defined ckpt')
            im_w, im_h=pretrained_dict['im_w'], pretrained_dict['im_h']
            print('Using resolution: {}x{}'.format(im_w,im_h))
            pretrained_dict=pretrained_dict['model']
        self.load_state_dict(pretrained_dict)
        print('Load model successfully')
        
    def load_pretrained(self, model_path=None):
        model_dict = self.state_dict()
        print('loading model from {}'.format(model_path))
        pretrained_dict = torch.load(model_path)
        keys=list(pretrained_dict.keys())
        if 'epoch' in keys:
            print('Restoring model from self-defined ckpt')
            pretrained_dict=pretrained_dict['model']
        tmp = OrderedDict()
        for k, v in pretrained_dict.items():
            if k in model_dict.keys():
                tmp[k] = v
            elif 'module' in k:  # multi_gpu
                t_k = k[k.find('.') + 1:]
                tmp[t_k] = v
        model_dict.update(tmp)
        self.load_state_dict(model_dict)
        print('Load model successfully')
 
    def init_module(self, module, init_type='norm', gain=0.01):
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                if init_type=='norm':
                    m.weight.data.normal_(0, gain)
                    m.bias.data.zero_()
                elif init_type=='kaiming':
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, gain)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def init_weights(self, pretrained=True):
        self.init_module(self.fpn, gain=cfg.GAIN)
        self.init_module(self.shared_conv, gain=cfg.GAIN)
        self.init_module(self.rpn_cls_conv, gain=cfg.GAIN)
        self.init_module(self.rpn_bbox_conv, gain=cfg.GAIN)
        self.init_module(self.fastRCNN, gain=cfg.GAIN)

    def make_net(self):
        self.padding=SamePad2d(kernel_size=3, stride=1)
        self.shared_conv=nn.Conv2d(self.features_out_ch,self.rpn_out_ch, 3, 1, padding=0)
        self.rpn_cls_conv=nn.Conv2d(self.rpn_out_ch, 2*self.K, 1, 1, padding=0)
        self.rpn_bbox_conv=nn.Conv2d(self.rpn_out_ch, 4*self.K, 1, 1, padding=0)        
        self.fastRCNN=FastRCNN(depth=self.features_out_ch, pool_size=self.frcnn_roi_size, num_classes=cfg.NUM_CLASSES)
        
    def get_params(self, lr=None):
        backbone_params=[]
        task_params=[]
        
        for k, value in self.named_parameters():
            if 'features' in k and value.requires_grad:
                backbone_params.append(value)
            elif value.requires_grad:
                task_params.append(value)
        params=[{'key':'backbone','params':backbone_params, 'lr':lr['backbone'], 'momentum':0.9},
                {'key':'task','params':task_params, 'lr':lr['task'], 'momentum':0.9}]
        return params
        
    def clip_boxes(self, boxes, bound):
        boxes[:,0]=np.minimum(bound[0],np.maximum(0, boxes[:,0]))
        boxes[:,1]=np.minimum(bound[1],np.maximum(0, boxes[:,1]))
        boxes[:,2]=np.maximum(0,np.minimum(bound[0], boxes[:,2]))
        boxes[:,3] = np.maximum(0, np.minimum(bound[1], boxes[:,3]))

    def gen_proposals(self, rpn_logits, rpn_bbox, anchors, batch_size):
        out_cls_softmax=F.softmax(rpn_logits, dim=1)
        out_cls_np=out_cls_softmax.cpu().data.numpy()[:,1,:,:]
        out_cls_np=out_cls_np.reshape(batch_size, self.K, self.out_height, self.out_width).transpose(0,2,3,1).reshape(-1, 1)
        '''
        out_cls=rpn_logits.view(batch_size, 2*self.K, self.out_height, self.out_width).permute(0,2,3,1).contiguous().view(-1,2)
        out_cls_softmax=F.softmax(out_cls, dim=1)
        out_cls_np=out_cls_softmax.cpu().data.numpy()[:,1].reshape(-1,1)
        '''
        out_bbox_np=rpn_bbox.cpu().data.numpy().transpose(0,2,3,1).reshape(-1, 4)
        tic=time()
        if cfg.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            out_bbox_np = out_bbox_np*cfg.RPN_BBOX_STD_DEV

        bbox_pred=bbox_transform_inv(anchors, out_bbox_np)
        self.clip_boxes(bbox_pred, self.bound)
        toc=time()
        if DEBUG:
            print('--bbox transform costs {}s'.format(toc-tic))
        bbox_pred_with_cls=np.hstack((bbox_pred, out_cls_np))
        proposals_batch=np.split(bbox_pred_with_cls, batch_size, axis=0)
        
        all_proposals=[]
        tic=time()
        for proposals in proposals_batch:
            '''
            x1,y1,x2,y2,_=np.split(proposals,5,axis=1)
            ws,hs=x2-x1+1,y2-y1+1
            inds=np.where(np.bitwise_and(ws>32,hs>32)==1)[0]
            proposals=proposals[inds]
            '''
            if cfg.NMS:
                order=np.argsort(proposals[:,-1])[::-1]
                proposals_order=proposals[order[:cfg[cfg.PHASE].RPN_PRE_NMS_TOP_N]]
                pick=nms_cuda(proposals_order, nms_thresh=cfg[cfg.PHASE].RPN_NMS_THRESH, xyxy=True)
                
                if len(pick)==0:
                    print('No pick in proposal nms')
                if cfg[cfg.PHASE].RPN_POST_NMS_TOP_N>0 and len(pick)>cfg[cfg.PHASE].RPN_POST_NMS_TOP_N:
                    pick=pick[:cfg[cfg.PHASE].RPN_POST_NMS_TOP_N]
                proposals_nms=proposals_order[pick]                
            else:
                proposals_nms=proposals
            all_proposals.append(proposals_nms)
        toc=time()
        if DEBUG:
            print('--NMS costs {}s'.format(toc-tic))
        '''list of all proposals of each image sample'''
        return all_proposals
    
    def get_rois(self, featuremap, proposals, num_proposals):
        assert featuremap.shape[0]==len(num_proposals)
        proposals=proposals/(1.0*self.stride)
        box_inds_np=[i*np.ones(num_proposals[i], dtype=np.int32) for i in range(len(num_proposals))]
        box_inds_np=np.concatenate(box_inds_np)
        box_inds=Variable(torch.from_numpy(box_inds_np).cuda())
        '''list'''
        if isinstance(proposals, list):
            proposals_all=np.vstack(proposals)
        else:
            proposals_all=proposals
        roi_features=crop_and_resize((self.frcnn_roi_size, self.frcnn_roi_size),featuremap, Variable(torch.from_numpy(proposals_all).float().cuda()), box_inds)
        return roi_features

    def fpn_out(self, x):
        c1, c2, c3, c4, c5=self.features(x)
        p2,p3,_,_,_=self.fpn(c1,c2,c3,c4,c5)

        if self.stride==8:
            return p3
        else:
            return p2
    
    def forward(self, roidbs):
        image_list=[]
        gt_boxes_list=[]
        anchor_list=[]
        gt_classes_list=[]
        for db in roidbs:
            if db['gt_boxes'] is not None:
                gt_boxes_list.append(db['gt_boxes'])
            if db['gt_classes'] is not None:
                gt_classes_list.append(db['gt_classes'])

            image_list.append(db['data'])
            anchor_list.append(db['anchors'])
         
        images = np.concatenate(image_list, axis=0)
        
        anchors=np.vstack(anchor_list)
        
        self.gt_boxes=gt_boxes_list
        self.gt_classes=gt_classes_list
        self.anchors=anchors
         #NHWC-->NCHW
        images=Variable(torch.from_numpy(images.transpose(0, 3, 1, 2)).cuda())
        tic=time()
        x=self.fpn_out(images)
        rpn_x = F.relu(self.shared_conv(self.padding(x)), inplace=True)    
        rpn_logits= self.rpn_cls_conv(rpn_x)
        rpn_logits = rpn_logits.view(self.batch_size, 2, self.K*self.out_height, self.out_width)
        rpn_bbox=self.rpn_bbox_conv(rpn_x)
        toc=time()
        if DEBUG:
            print('Features cost {}s'.format(toc-tic))
        tic=time()
        '''Fast RCNN start'''
        all_proposals=self.gen_proposals(rpn_logits, rpn_bbox, anchors, self.batch_size)
        toc=time()
        if DEBUG:
            print('Gen proposals cost {}s'.format(toc-tic))
        if cfg.PHASE=='TRAIN':
            proposals, proposal_cls_targets, proposal_bbox_targets, bbox_weights, labels=\
                get_proposal_target(all_proposals, self.gt_boxes, self.gt_classes, self.batch_size)
            num_proposals_per_image=[]
            num_fgs=0
            for label in labels:
                num_proposals_per_image.append(len(label))
                fg_inds=np.where(label>0)[0]
                num_fgs+=len(fg_inds)
            tic=time()
            roi_features=self.get_rois(x, proposals, num_proposals_per_image)
            frcnn_logits, frcnn_probs, frcnn_bbox=self.fastRCNN(roi_features)
            toc=time()
            if DEBUG:
                print('RoIAlign and FastRCNN cost {}s'.format(toc-tic))
            frcnn_bbox=torch.mul(frcnn_bbox, Variable(torch.from_numpy(bbox_weights).cuda(), requires_grad=False))
        else:
            num_proposals_per_image=[]
            '''
            if nms, proposals may not have the same number of each box
            if not nms, each box has same number of proposals
            '''
            for prop in all_proposals:
                num_proposals_per_image.append(prop.shape[0])
            proposals=np.vstack(all_proposals)
            roi_features=self.get_rois(x, proposals, [len(proposals)])
            frcnn_logits, frcnn_probs, frcnn_bbox=self.fastRCNN(roi_features)
        
        '''Fast-RCNN end'''
        output={}
        output['rpn_logits']=rpn_logits
        output['rpn_bbox']=rpn_bbox
        output['frcnn_bbox']=frcnn_bbox
        
        if cfg.PHASE=='TRAIN':
            output['frcnn_logits_target']=Variable(torch.from_numpy(proposal_cls_targets).long().cuda())
            output['frcnn_bbox_target']=Variable(torch.from_numpy(proposal_bbox_targets).float().cuda())
            output['num_proposals']=num_proposals_per_image
            output['frcnn_logits']=frcnn_logits
            output['num_fgs']=num_fgs
            output['labels']=labels
        else:
            output['frcnn_probs']=frcnn_probs
            output['proposals']=proposals
            output['num_proposals']=num_proposals_per_image
#        return out_cls, out_bbox, all_proposals
        return output
    
class SamePad2d(nn.Module):
    """Mimics tensorflow's 'SAME' padding.
    """

    def __init__(self, kernel_size, stride):
        super(SamePad2d, self).__init__()
        self.kernel_size = torch.nn.modules.utils._pair(kernel_size)
        self.stride = torch.nn.modules.utils._pair(stride)

    def forward(self, input):
        in_width = input.size()[2]
        in_height = input.size()[3]
        out_width = math.ceil(float(in_width) / float(self.stride[0]))
        out_height = math.ceil(float(in_height) / float(self.stride[1]))
        pad_along_width = ((out_width - 1) * self.stride[0] +
                           self.kernel_size[0] - in_width)
        pad_along_height = ((out_height - 1) * self.stride[1] +
                            self.kernel_size[1] - in_height)
        pad_left = math.floor(pad_along_width / 2)
        pad_top = math.floor(pad_along_height / 2)
        pad_right = pad_along_width - pad_left
        pad_bottom = pad_along_height - pad_top
        return F.pad(input, (pad_left, pad_right, pad_top, pad_bottom), 'constant', 0)

    def __repr__(self):
        return self.__class__.__name__

if __name__=='__main__':
    model=FasterRCNN(100,100)
    torch.save(model.state_dict(), 'faster_rcnn.pkl')
    params=model.get_params({'backbone':0.1, 'task':0.5})