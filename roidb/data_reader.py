import numpy as np
import cv2
import rpn.generate_anchors as G
from core.config import cfg

class DataReader(object):
    def __init__(self, im_width, im_height, batch_size=8):
        self.im_w = im_width
        self.im_h = im_height
        self.batch_size=batch_size   
        self.stride=cfg.STRIDE

        self.bound=(im_width, im_height)
        self.out_size=(self.im_w//self.stride, self.im_h//self.stride)

        self.fetch_config()     

        self.raw_anchors=G.generate_anchors(self.basic_size, self.ratios, self.scales)
        dummy_search_box=np.array([[0,0,self.im_w-1,self.im_h-1]])
        self.anchors=G.gen_region_anchors(self.raw_anchors, \
            dummy_search_box, self.bound, K=self.K, size=self.out_size)[0]

    def fetch_config(self):
        self.stride=cfg.STRIDE
        self.basic_size=cfg.BASIC_SIZE
        self.ratios=cfg.RATIOS
        self.scales=cfg.SCALES
        
        self.K=len(self.ratios)*len(self.scales)

    def filter_boxes(self, boxes):
        x1,y1,x2,y2=np.split(boxes, 4, axis=1)

        ws=x2-x1+1
        hs=y2-y1+1

        filter_inds=np.where(np.bitwise_and(ws>16,hs>16)==1)[0]
        return filter_inds

    def __len__(self):
         return self.num_samples
         
    def __getitem__(self, index):
        roidb=self._get_roidb(index)
        return roidb

    def flip(self, image, boxes):
        image_flip=cv2.flip(image, 1)
        boxes_flip=boxes.copy()
        h,w=image.shape[:2]
        boxes_flip[:,0]=w-1-boxes[:,2]
        boxes_flip[:,2]=w-1-boxes[:,0]

        return image_flip, boxes_flip

    def _get_roidb(self, index):
        raise NotImplementedError

    def _parse_all_anno(self):
        raise NotImplementedError