import torch
import torch.nn as nn
from collections import OrderedDict
import os.path as op
from core.config import cfg


class Vgg16(nn.Module):
    def __init__(self, bn=True, relu=False, pretrained=False):
        super(Vgg16, self).__init__()
        self.bn=bn
        self.relu=relu  #relu or not after the last layer
        self.out_ch=0
        self.make_features()

        VGG_PRETRAINED_BN=op.join(cfg.PRETRAINED_DIR, 'vgg16_bn.pth')
        VGG_PRETRAINED=op.join(cfg.PRETRAINED_DIR, 'vgg16.pth')
        if pretrained:
            model_path=VGG_PRETRAINED_BN if self.bn else VGG_PRETRAINED
            self.load_pretrained(model_path=model_path)
        
    def make_features(self):
        vgg_layers=[[64, 64, 'M'],
            [128, 128, 'M'], 
            [256, 256, 256, 'M'], 
            [512, 512, 512]]
        
        self.conv1=self._make_layer(3, vgg_layers[0])
        self.conv2=self._make_layer(64, vgg_layers[1])
        self.conv3=self._make_layer(128, vgg_layers[2])
        self.conv4=self._make_layer(256, vgg_layers[3])
        
    def _make_layer(self, in_ch, channels, relu=True):
        layers=[]
        for i, _chn in enumerate(channels):
            if _chn=='M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(in_ch, _chn, 3, stride=1, padding=1))
                if self.bn:
                    layers.append(nn.BatchNorm2d(_chn))
                if i<len(channels)-1:
                    layers.append(nn.ReLU(inplace=True))
                elif relu:
                    layers.append(nn.ReLU(inplace=True))
            in_ch=_chn
        return nn.Sequential(*layers)

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

    def init_weights(self, init_type='norm', gain=0.01):
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
        
    def forward(self, x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        return x
    
if __name__=='__main__':
    model=Vgg16()
    model_path='/home/yfji/Pretrained/pytorch/vgg16_bn.pth'
    state_dict=model.state_dict()
    torch.save(state_dict, 'vgg16.pkl')
    for k, v in state_dict.items():
        print(k,v.shape)
    model.init_weights()
    model.load_weights(model_path=model_path)