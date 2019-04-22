from .data_reader import DataReader
import numpy as np
import cv2
import os
import os.path as op
#import shutil
import rpn.util as U
import xml.etree.ElementTree as ET
from core.config import cfg
import copy

EXTRA_SEQS=['MVI_39761','MVI_39781','MVI_39811','MVI_39851','MVI_39931','MVI_40152','MVI_40162','MVI_40211','MVI_40213','MVI_40991','MVI_40992','MVI_63544']
CAT_IND_MAP={'car':1,'van':2,'bus':3,'truck':4}

class DetracDataReader(DataReader):
    def __init__(self, im_width, im_height, batch_size=8):
        super(DetracDataLoader, self).__init__(im_width, im_height, batch_size)
        self.data_dir = '/mnt/sda7/DETRAC-train-data/Insight-MVT_Annotation_Train'
        self.anno_dir = '/mnt/sda7/DETRAC-train-data/DETRAC-Train-Annotations-XML'
        
        self.img_dirs=sorted(os.listdir(self.data_dir))
        self.anno_files=sorted(os.listdir(self.anno_dir))

        for ext_seq in EXTRA_SEQS:
            ext_anno='{}.xml'.format(ext_seq)
            self.anno_files.remove(ext_anno)
            self.img_dirs.remove(ext_seq)
        
        self.num_samples=0
        self.num_visualize = 100

        self.enum_sequences()
        self._parse_all_anno()
        
    def enum_sequences(self):
        self.images_per_seq=np.zeros(self.num_sequences, dtype=np.int32)
        self.index_per_seq=np.zeros(self.num_sequences, dtype=np.int32)
        self.start_per_seq=np.zeros(self.num_sequences, dtype=np.int32)
        for i in range(self.num_sequences):
            img_dir=self.img_dirs[i]
            num_samples_this_dir=len(os.listdir(op.join(self.data_dir,img_dir)))
            self.images_per_seq[i]=num_samples_this_dir
            self.start_per_seq[i]=self.num_samples
            self.num_samples+=num_samples_this_dir

    def _parse_all_anno(self):
        print('Preparing dataset...')
        all_anno=[]
        all_imgs=[]
        for i in range(self.num_sequences):
            img_dir=op.join(self.data_dir, self.img_dirs[i])
            anno_file=op.join(self.anno_dir, self.anno_files[i])
            tree = ET.parse(anno_file)
            root = tree.getroot()

            frames=root.findall('frame')

            img_files=os.listdir(img_dir)
            for j, frame in enumerate(frames):
                target_list=frame.find('target_list')
                targets=target_list.findall('target')

                all_imgs.append(op.join(self.img_dirs[i], img_files[j]))
                all_anno.append(targets)

        assert self.num_samples==len(all_anno), 'Annotations: {} and num_images: {}'.format(len(all_anno), self.num_images)
        print('Dataset is available')
        self.dataset=all_imgs
        self.annotations=all_anno

    def __len__(self):
        return self.num_samples
    
    def _get_roidb(self, index):          
        img_path=op.join(self.data_dir, self.dataset[index])
        targets=self.annotations[index]

        roidb={}
        gt_boxes=np.zeros((0,4),dtype=np.float32)
        gt_classes=np.zeros(0, dtype=np.int32)

        image = cv2.imread(img_path)
        
        h,w=image.shape[0:2]
        image=cv2.resize(image, (self.im_w, self.im_h), interpolation=cv2.INTER_LINEAR)
        nh, nw=image.shape[0:2]

        yscale=1.0*nh/h
        xscale=1.0*nw/w

        for obj in targets:
            bbox = np.zeros(4, dtype=np.float32)

            bbox_attribs=obj.find('box').attrib
            attribute_attribs=obj.find('attribute').attrib

            left=float(bbox_attribs['left'])
            top=float(bbox_attribs['top'])
            width=float(bbox_attribs['width'])
            height=float(bbox_attribs['height'])

            bbox[0]=left
            bbox[1]=top
            bbox[2]=left+width-1
            bbox[3]=top+height-1
      
            bbox[[0,2]]*=xscale
            bbox[[1,3]]*=yscale
            cat=attribute_attribs['vehicle_type']
            if cat=='others':
                cat='truck'
            cat_ind=CAT_IND_MAP[cat]

            gt_boxes=np.append(gt_boxes, bbox.reshape(-1,4), axis=0)
            gt_classes=np.append(gt_classes, cat_ind)

        flip_prob=np.random.rand()
        if flip_prob>0.5:
            image, gt_boxes=self.flip(image, gt_boxes)
        roidb['data']=image[np.newaxis, :,:,:].astype(np.float32)
        roidb['gt_boxes']=gt_boxes
        roidb['gt_classes']=gt_classes

        roidb['bound']=self.bound

        bbox_overlaps=U.bbox_overlaps_per_image(self.anchors, gt_boxes)
        
        roidb['anchors']=self.anchors
        roidb['bbox_overlaps']=bbox_overlaps

        return roidb