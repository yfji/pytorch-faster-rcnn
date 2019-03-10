import torch
import torch.nn as nn

class RPN(nn.Module):
    def __init__(self, in_ch, out_ch, K=9):
        super(RPN, self).__init__()
        self.in_channel=in_ch
        self.K=K
    
        self.out_ch=out_ch
        self.make_rpn()
        
    def make_rpn(self):
        self.rpn_conv=nn.Conv2d(self.in_channel, self.out_ch, 3, 1, padding=1)
        self.cls_conv=nn.Conv2d(self.out_ch, 2*self.K*self.out_ch, 3, 1, padding=1)
        self.bbox_conv=nn.Conv2d(self.out_ch, 4*self.K*self.out_ch, 3, 1, padding=1)
    
    def forward(self, in_tensor):
#        out_tensor=in_tensor
        out_tensor=self.rpn_conv(in_tensor)
#        out_tensor=F.relu(out_tensor, inplace=True)
        cls_tensor=self.cls_conv(out_tensor)
        bbox_tensor=self.bbox_conv(out_tensor)
        
        return cls_tensor, bbox_tensor