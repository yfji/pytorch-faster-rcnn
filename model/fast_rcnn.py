import torch.nn as nn
import torch.nn.functional as F

class FastRCNN(nn.Module):
    def __init__(self, depth=256, pool_size=7, num_classes=81):
        super(FastRCNN, self).__init__()
        self.depth=depth
        self.pool_size=pool_size
        self.num_classes=num_classes
        self.num_bbox_output=self.num_classes*4
        self.channel=512
#        self.drop_prob=0.5
        self.make_fast_rcnn()
        
    def make_fast_rcnn(self):
        self.conv1=nn.Conv2d(self.depth, self.channel, kernel_size=self.pool_size, stride=1)
        self.bn1=nn.BatchNorm2d(self.channel,eps=0.001,momentum=0.01)
        self.conv2=nn.Conv2d(self.channel,self.channel,kernel_size=1,stride=1)
        self.bn2=nn.BatchNorm2d(self.channel,eps=0.001,momentum=0.01)
        self.relu=nn.ReLU(inplace=True)

        self.linear_class=nn.Linear(self.channel, self.num_classes)
        self.linear_bbox=nn.Linear(self.channel, self.num_bbox_output)

        self.softmax=nn.Softmax(dim=1)
    
    def forward(self, in_tensor):
        assert in_tensor.shape[2]==self.pool_size and in_tensor.shape[3]==self.pool_size

        x=self.conv1(in_tensor)
        x=self.bn1(x)
        x=F.relu(x, inplace=True)
        x=self.conv2(x)
        x=self.bn2(x)
        x=F.relu(x, inplace=True)

        x=x.view(-1,self.channel)

        frcnn_logits=self.linear_class(x)
        frcnn_probs=self.softmax(frcnn_logits)

        frcnn_bbox=self.linear_bbox(x)

        return frcnn_logits, frcnn_probs, frcnn_bbox
