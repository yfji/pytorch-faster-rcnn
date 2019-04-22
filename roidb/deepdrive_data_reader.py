from .data_reader import DataReader
import numpy as np
import cv2
import os
import os.path as op
#import shutil
import rpn.util as U

import roidb.image_utils as util
from core.config import cfg
import copy
import json

CLASSES=['__background__','bike', 
                        'bus', 
                        'car',
                        'motor', 
                        'person', 
                        'rider', 
                        'traffic light', 
                        'traffic sign', 
                        'train', 
                        'truck']


CAT_IND_MAP={CLASSES[i]:i for i in range(1, len(CLASSES))}

class DeepDriveDataReader(DataReader):
    def __init__(self, im_width, im_height, batch_size=8):
        super(DeepDriveDataReader, self).__init__(im_width, im_height, batch_size)
        self.label_path='/mnt/sda7/DeepDrive/bdd100k/labels/bdd100k_labels_images_train.json'
        self.image_directory='/mnt/sda7/DeepDrive/bdd100k/images/100k/train'
        
        self.num_samples=0
        
        self.num_visualize = 100        

        self._parse_all_anno()

        self.permute_inds = np.random.permutation(np.arange(self.num_images))
        cfg.NUM_CLASSES=len(CLASSES)
       

    def _parse_all_anno(self):
        self.dataset=json.load(open(self.label_path, 'r'))
        self.num_samples=len(self.dataset)
        self.upper_bound=self.num_images-self.batch_size

    def __len__(self):
        return self.num_samples

    def filter_boxes(self, boxes):
        x1,y1,x2,y2=np.split(boxes, 4, axis=1)

        ws=x2-x1+1
        hs=y2-y1+1

        filter_inds=np.where(np.bitwise_and(ws>16,hs>16)==1)[0]
        return filter_inds

    def _get_roidb(self, index):
        item=self.dataset[index]
        file_name=item['name']
        im_path=os.path.join(self.image_directory, file_name)

        image=cv2.imread(im_path)
        h,w=image.shape[0:2]
        image=cv2.resize(image, (self.im_w, self.im_h), interpolation=cv2.INTER_LINEAR)
        nh, nw=image.shape[0:2]

        yscale=1.0*nh/h
        xscale=1.0*nw/w

        labels=item['labels']

        gt_boxes=np.zeros((0,4),dtype=np.float32)
        gt_classes = np.zeros(0, dtype=np.int32)
        
        for inst in labels:
            cat=inst['category']  
            if cat!='drivable area' and cat!='lane':
                box=inst['box2d']
                x1, y1, x2, y2=float(box['x1'])*xscale, float(box['y1'])*yscale, float(box['x2'])*xscale, float(box['y2'])*yscale
                gt_box=np.asarray([[x1,y1,x2,y2]])
                gt_ind=CAT_IND_MAP[cat]

                gt_boxes=np.append(gt_boxes, gt_box, 0)
                gt_classes=np.append(gt_classes, gt_ind)
        
        flip_prob=np.random.rand()
        if flip_prob>0.5:
            image, gt_boxes=self.flip(image, gt_boxes)

        roidb={}
        roidb['data']=image[np.newaxis, :,:,:].astype(np.float32)
        roidb['gt_boxes']=gt_boxes
        
        roidb['gt_classes']=gt_classes

        roidb['bound']=self.bound
        bbox_overlaps=U.bbox_overlaps_per_image(self.anchors, gt_boxes)
        
        roidb['anchors']=self.anchors
        roidb['bbox_overlaps']=bbox_overlaps
 
        return roidb