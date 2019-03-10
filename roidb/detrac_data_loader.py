import numpy as np
import cv2
import os
import os.path as op
#import shutil
import rpn.util as U
import rpn.generate_anchors as G
import roidb.image_utils as util
import xml.etree.ElementTree as ET
from core.config import cfg
import copy

MAX_SEQ_LEN=-1
EXTRA_SEQS=['MVI_39761','MVI_39781','MVI_39811','MVI_39851','MVI_39931','MVI_40152','MVI_40162','MVI_40211','MVI_40213','MVI_40991','MVI_40992','MVI_63544']
CAT_IND_MAP={'car':1,'van':2,'bus':3,'truck':4}

class DetracDataLoader(object):
    def __init__(self, im_width, im_height, batch_size=8):
        self.data_dir = '/mnt/sda7/DETRAC-train-data/Insight-MVT_Annotation_Train'
        self.anno_dir = '/mnt/sda7/DETRAC-train-data/DETRAC-Train-Annotations-XML'
        
        self.stride=cfg.STRIDE
        self.basic_size=cfg.BASIC_SIZE
        self.ratios=cfg.RATIOS
        self.scales=cfg.SCALES
        
        self.K=len(self.ratios)*len(self.scales)
        
        self.img_dirs=sorted(os.listdir(self.data_dir))
        self.anno_files=sorted(os.listdir(self.anno_dir))

        for ext_seq in EXTRA_SEQS:
            ext_anno='{}.xml'.format(ext_seq)
#            assert ext_anno in self.anno_files, '{} not exists'.format(ext_anno)
            self.anno_files.remove(ext_anno)
            self.img_dirs.remove(ext_seq)

        self.index = 0
        self.vis_dir = './vis_vid'
        self.vis_index = 0

        self.im_w = im_width
        self.im_h = im_height
        
        self.batch_size=batch_size
        self.out_size=(self.im_w//self.stride, self.im_h//self.stride)

        self.num_sequences = len(self.anno_files)
        self.num_images=0
        
        self.num_visualize = 100
        self.permute_inds = np.random.permutation(np.arange(self.num_sequences))

        self.iter_stop=False
        self.enum_sequences()
        
        self.raw_anchors=G.generate_anchors(self.basic_size, self.ratios, self.scales)

    def gen_anchors(self, search_boxes, bound):
        box_anchors=[]
        A=self.out_size[0]*self.out_size[1]
        K=self.K
        for i in range(len(search_boxes)):
            shifts_ctrs = G.calc_roi_align_shifts(search_boxes[i], self.out_size, bound)
            anchors = self.raw_anchors.reshape((1, K, 4)) + shifts_ctrs.reshape((1, A, 4)).transpose((1, 0, 2))
            anchors = anchors.reshape((A*K, 4))
            box_anchors.append(anchors)
        return box_anchors
    
    def enum_sequences(self):
        self.images_per_seq=np.zeros(self.num_sequences, dtype=np.int32)
        self.index_per_seq=np.zeros(self.num_sequences, dtype=np.int32)
        self.inds_per_seq=[]
        for i in range(self.num_sequences):
            img_dir=self.img_dirs[i]
            num_samples_this_dir=len(os.listdir(op.join(self.data_dir,img_dir)))
            self.images_per_seq[i]=min(num_samples_this_dir, MAX_SEQ_LEN if MAX_SEQ_LEN>0 else num_samples_this_dir)
            self.num_images+=num_samples_this_dir
            self.inds_per_seq.append(np.random.permutation(np.arange(self.images_per_seq[i])))

        self.upper_bound_per_seq=self.images_per_seq-self.batch_size

    def get_num_samples(self):
        return np.sum(self.images_per_seq)

    def filter_boxes(self, boxes):
        x1,y1,x2,y2=np.split(boxes, 4, axis=1)

        ws=x2-x1+1
        hs=y2-y1+1

        filter_inds=np.where(np.bitwise_and(ws>16,hs>16)==1)[0]
        return filter_inds
    
    def get_minibatch(self):
        return self.get_minibatch_inter_img()

    def get_minibatch_inter_img(self):
        if self.iter_stop:
            self.iter_stop=False
            return None
        
        roidbs=[]
        index = self.permute_inds[self.index]
        while self.index_per_seq[index]>self.upper_bound_per_seq[index]:
            self.index+=1
            if self.index==self.num_sequences:
                self.index=0
            index=self.permute_inds[self.index]
            
        anno_file=op.join(self.anno_dir, self.anno_files[index])
        img_dir=op.join(self.data_dir, self.img_dirs[index])

        img_files=sorted(os.listdir(img_dir))
        tree = ET.parse(anno_file)
        root = tree.getroot()

        frames=root.findall('frame')

        cur_image_index=self.index_per_seq[index]
        image_inds=[self.inds_per_seq[index][cur_image_index+i] for i in range(self.batch_size)]
                
        for ind in image_inds:
            roidb={}
            gt_boxes=np.zeros((0,4),dtype=np.float32)
            gt_classes=np.zeros(0, dtype=np.int32)

            img_file=img_files[ind]
            frame=frames[ind]
            
            image = cv2.imread(op.join(img_dir, img_file))
            
            h,w=image.shape[0:2]
            image=cv2.resize(image, (self.im_w, self.im_h), interpolation=cv2.INTER_LINEAR)
            nh, nw=image.shape[0:2]

            yscale=1.0*nh/h
            xscale=1.0*nw/w

            target_list=frame.find('target_list')
            targets=target_list.findall('target')
                        
            for obj in targets:
                attribs=obj.attrib
                obj_id = int(attribs['id'])

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
                '''
                bbox*=scale
                bbox[[0,2]]+=xstart
                bbox[[1,3]]+=ystart
                '''
                bbox[[0,2]]*=xscale
                bbox[[1,3]]*=yscale
                cat=attribute_attribs['vehicle_type']
                if cat=='others':
                    cat='truck'
                cat_ind=CAT_IND_MAP[cat]

                gt_boxes=np.append(gt_boxes, bbox.reshape(-1,4), axis=0)
                gt_classes=np.append(gt_classes, cat_ind)

            roidb['image']=image[np.newaxis, :,:,:].astype(np.float32)
            roidb['gt_boxes']=gt_boxes
           
            roidb['gt_classes']=gt_classes

            bound=(image.shape[1], image.shape[0])
            roidb['bound']=bound

            dummy_search_box=np.array([[0,0,self.im_w-1,self.im_h-1]])
            anchors=G.gen_region_anchors(self.raw_anchors, dummy_search_box, bound, K=self.K, size=self.out_size)[0]

#            print(anchors.shape)
            
            bbox_overlaps=U.bbox_overlaps_per_image(anchors, gt_boxes, branch='frcnn')
            
            roidb['anchors']=anchors
            roidb['bbox_overlaps']=bbox_overlaps
            roidbs.append(roidb)

        self.index_per_seq[index] += self.batch_size
        index_res=self.index_per_seq-self.upper_bound_per_seq
        index_res=index_res[self.permute_inds]
        valid_seq_inds=np.where(index_res<=0)[0]
        if valid_seq_inds.size==0:
            self.index_per_seq=np.zeros(self.num_sequences, dtype=np.int32)
            self.inds_per_seq=[np.random.permutation(np.arange(self.images_per_seq[i])) for i in range(self.num_sequences)]
            self.permute_inds = np.random.permutation(np.arange(self.num_sequences))
            self.index=0
            self.iter_stop=True
        else:    
            self.index+=1
            if self.index==self.num_sequences:
                self.index=0
        
        return roidbs