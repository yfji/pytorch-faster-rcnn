import sys
sys.path.insert(0,'/home/yfji/Workspace/PyTorch/DLMOT-FRCNN/nms')
from nms_wrapper import nms
import numpy as numpy
import torch
import numpy as np

bboxes=np.array([[0,0,50,50],
                [1,1,51,51],
                [2,2,52,52],
                [25,25,75,75],
                [26,26,76,76],
                [100,100,150,150],
                [101,101,151,151]])

scores=np.array([0.9,0.8,0.7,0.9,0.5,0.2,0.6])

bboxes_with_score=np.hstack((bboxes, scores.reshape(-1,1)))
bboxes_with_score[:,0]+=20

bboxes_pth=torch.from_numpy(bboxes_with_score).float().cuda()

keep=nms(bboxes_pth, 0.7)
print(keep)