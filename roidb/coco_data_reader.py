from .data_reader import DataReader
import json
import numpy as np
import os.path as op
from collections import OrderedDict
import cv2
import rpn.util as U
from core.config import cfg

coco_id_name_map={0:'__background__',1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
                   6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
                   11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
                   16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow',
                   22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
                   28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
                   35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat',
                   40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket',
                   44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
                   51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
                   56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
                   61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table',
                   70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard',
                   77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink',
                   82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors',
                   88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}

class COCODataReader(DataReader):
    def __init__(self, im_width, im_height, batch_size=8):
        super(COCODataReader, self).__init__(im_width, im_height, batch_size=batch_size)
        self.json_path='/mnt/sda7/MSCOCO/annotations/coco_images_train2014.json'
        self.image_dir='/mnt/sda7/MSCOCO/train2014'

        self._parse_all_anno()

        self.ordered_id_name_map=sorted(zip(coco_id_name_map.keys(),coco_id_name_map.values()), key=lambda x:x[0])
        self.coco_class_id_map=OrderedDict()
        for i, tup in enumerate(self.ordered_id_name_map):
            self.coco_class_id_map[tup[1]]=i
        self.num_classes=len(self.ordered_id_name_map)
        print('COCO dataset has classes: {}'.format(self.num_classes))
        cfg.NUM_CLASSES=self.num_classes
        self.num_samples=len(self.annotations)

    def __len__(self):
        return self.num_samples

    def _parse_all_anno(self):
        print('Preparing dataset...')
        with open(self.json_path, 'r') as f:
            dataset=json.load(f)
        self.annotations=dataset['annotations']
        print('Dataset is available. {} samples loaded'.format(len(self.annotations)))

    def _get_roidb(self, index):
        roidb={}
        sample=self.annotations[index]
        image_id=sample[0]['image_id']
        img_path=op.join(self.image_dir, 'COCO_train2014_%012d.jpg'%image_id)
        image=cv2.imread(img_path)
        if image is None:
            print('{} does not exists!'.format(img_path))

        fx=self.im_w*1.0/image.shape[1]
        fy=self.im_h*1.0/image.shape[0]

        image=cv2.resize(image, (self.im_w, self.im_h), interpolation=cv2.INTER_LINEAR)
        
        gt_boxes=np.zeros((0,4),dtype=np.float32)
        gt_classes=np.zeros(0, dtype=np.int32)

        for entry in sample:
            bbox=np.asarray(entry['bbox']).astype(np.float32)
            bbox[2]+=bbox[0]
            bbox[3]+=bbox[1]
            
            bbox[[0,2]]*=fx
            bbox[[1,3]]*=fy
                
            cat_id=entry['category_id']
            class_name=coco_id_name_map[cat_id]
            cls_id=self.coco_class_id_map[class_name]
            gt_boxes=np.append(gt_boxes, bbox.reshape(1,4), 0)            
            gt_classes=np.append(gt_classes, cls_id)

        flip_prob=np.random.rand()
        if flip_prob>0.5:
            image, gt_boxes=self.flip(image, gt_boxes)
        
        roidb['data']=image[np.newaxis,:,:,:].astype(np.float32)
        roidb['gt_boxes']=gt_boxes
        roidb['gt_classes']=gt_classes

        roidb['bound']=self.bound
        bbox_overlaps=U.bbox_overlaps_per_image(self.anchors, gt_boxes)
        
        roidb['anchors']=self.anchors
        roidb['bbox_overlaps']=bbox_overlaps

        return roidb