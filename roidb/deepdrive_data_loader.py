import numpy as np
import cv2
import os
import os.path as op
#import shutil
import rpn.util as U
import rpn.generate_anchors as G
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

cfg.NUM_CLASSES=len(CLASSES)

CAT_IND_MAP={CLASSES[i]:i for i in range(1, len(CLASSES))}
print(CAT_IND_MAP)

class DeepDriveDataLoader(object):
    def __init__(self, im_width, im_height, batch_size=8):
        self.label_path='/mnt/sda7/DeepDrive/bdd100k/labels/bdd100k_labels_images_train.json'
        self.image_directory='/mnt/sda7/DeepDrive/bdd100k/images/100k/train'

        self.stride=cfg.STRIDE
        self.basic_size=cfg.BASIC_SIZE
        self.ratios=cfg.RATIOS
        self.scales=cfg.SCALES
        
        self.K=len(self.ratios)*len(self.scales)
        
        self.index = 0

        self.im_w = im_width
        self.im_h = im_height
        
        self.batch_size=batch_size
        self.out_size=(self.im_w//self.stride, self.im_h//self.stride)

        self.num_images=0
        
        self.num_visualize = 100        

        self.iter_stop=False
        self.enum_dataset()

        self.permute_inds = np.random.permutation(np.arange(self.num_images))
        
        self.raw_anchors=G.generate_anchors(self.basic_size, self.ratios, self.scales)

    def enum_dataset(self):
        self.dataset=json.load(open(self.label_path, 'r'))
        self.num_images=len(self.dataset)
        self.upper_bound=self.num_images-self.batch_size

    def get_num_samples(self):
        return self.num_images

    def filter_boxes(self, boxes):
        x1,y1,x2,y2=np.split(boxes, 4, axis=1)

        ws=x2-x1+1
        hs=y2-y1+1

        filter_inds=np.where(np.bitwise_and(ws>16,hs>16)==1)[0]
        return filter_inds

    def get_minibatch(self):
        if self.iter_stop:
            self.iter_stop=False
            return None
        
        roidbs=[]

        perm_inds=self.permute_inds[self.index:self.index+self.batch_size]
        
        for ind in perm_inds:
            item=self.dataset[ind]
            file_name=item['name']
            im_path=os.path.join(self.image_directory, file_name)

            image=cv2.imread(im_path)
            h,w=image.shape[0:2]
            image=cv2.resize(image, (self.im_w, self.im_h), interpolation=cv2.INTER_LINEAR)
            nh, nw=image.shape[0:2]

            yscale=1.0*nh/h
            xscale=1.0*nw/w

            labels=item['labels']
#            num_instances=len(labels)

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
            
            roidb={}
            roidb['image']=image[np.newaxis, :,:,:].astype(np.float32)
            roidb['gt_boxes']=gt_boxes
           
            roidb['gt_classes']=gt_classes

            bound=(image.shape[1], image.shape[0])
            roidb['bound']=bound

            dummy_search_box=np.array([[0,0,self.im_w-1,self.im_h-1]])
            anchors=G.gen_region_anchors(self.raw_anchors, dummy_search_box, bound, K=self.K, size=self.out_size)[0]

            bbox_overlaps=U.bbox_overlaps_per_image(anchors, gt_boxes, branch='frcnn')
            
            roidb['anchors']=anchors
            roidb['bbox_overlaps']=bbox_overlaps
            roidbs.append(roidb)
        
        self.index+=self.batch_size
        if self.index>self.upper_bound:
            self.index=0
            self.permute_inds = np.random.permutation(np.arange(self.num_images))
            self.iter_stop=True
        
        return roidbs