import torch
import torch.nn as nn
from core.config import cfg
from collections import OrderedDict

class Vgg16(nn.Module):
    def __init__(self, bn=True, relu=False):
        super(Vgg16, self).__init__()
        self.bn=bn
        self.relu=relu
        self.out_ch=0
        self.make_features()
        
    def make_features(self):
        vgg_layers=[[3,0,64],
                    [3,2,64],
                    [3,0,128],
                    [3,2,128],
                    [3,0,256],
                    [3,0,256],
                    [3,2,256],
                    [3,0,512],
                    [3,0,512]]
#                    [3,2,512],
#                    [3,0,512],
#                    [3,0,512],
#                    [3,0,512]
#                    ]
        adjust_layers=[[3,0,256],
                       [3,0,256],
                       [1,0,256]]
        
        features=[]
        adjust=[]
        in_ch=3
        for i in range(len(vgg_layers)):
            ksizes=vgg_layers[i]
            kernel_size=ksizes[0]
            pool_size=ksizes[1]
            out_channel=ksizes[2]
            self.out_ch=out_channel
            features.append(nn.Conv2d(in_ch, out_channel, kernel_size, stride=1, padding=kernel_size//2))
            if self.bn:
                features.append(nn.BatchNorm2d(out_channel))
            features.append(nn.ReLU(inplace=True))
            if pool_size>0:
                features.append(nn.MaxPool2d(pool_size))
            in_ch=out_channel
        self.features=nn.Sequential(*features)
        
        for i in range(len(adjust_layers)):
            ksizes=adjust_layers[i]
            kernel_size=ksizes[0]
            out_channel=ksizes[2]
            self.out_ch=out_channel
            adjust.append(nn.Conv2d(in_ch, out_channel, kernel_size, stride=1, padding=kernel_size//2))
            if self.bn:
                adjust.append(nn.BatchNorm2d(out_channel))
            if i<len(adjust_layers)-1:
                adjust.append(nn.ReLU(inplace=True))
            in_ch=out_channel
        if self.relu:
            adjust.append(nn.ReLU(inplace=True))
        self.adjusts=nn.Sequential(*adjust)
        
    def load_pretrained(self, model_path=None):
        model_dict = self.state_dict()
        print('loading model from {}'.format(model_path))
        pretrained_dict = torch.load(model_path)
        tmp = OrderedDict()
        for k, v in pretrained_dict.items():
            if k in model_dict.keys():
                tmp[k] = v
            elif 'module' in k:  # multi-gpu
                t_k = k[k.find('.') + 1:]
                tmp[t_k] = v
        model_dict.update(tmp)
        self.load_state_dict(model_dict)
        print('Model updated successfully')

    def init_weights(self, init_type='norm', gain=cfg.GAIN):
        for m in self.modules():
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
        
    def forward(self, in_tensor):
        feature=self.features(in_tensor)
        adjust=self.adjusts(feature)
        return adjust
    
if __name__=='__main__':
    model=Vgg16()
    model_path='/home/yfji/Pretrained/pytorch/vgg16_bn.pth'
    state_dict=model.state_dict()
    torch.save(state_dict, 'vgg16.pkl')
    for k, v in state_dict.items():
        print(k,v.shape)
    model.init_weights()
    model.load_weights(model_path=model_path)
    
        
