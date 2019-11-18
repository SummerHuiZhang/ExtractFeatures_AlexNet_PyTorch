import math
import itertools
import datetime
import time
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

#from models import *
from datasets import *
#from utils import *

import torch.nn as nn
import torch.nn.functional as F
import torch
import sys
sys.path.append('..')
import torch.utils.model_zoo as model_zoo
import cv2
import numpy as np
from PIL import Image
from sklearn.decomposition import IncrementalPCA
import matplotlib.pyplot as plt
import glob
from collections import OrderedDict
__all__ = ['AlexNet', 'alexnet']


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
        #self.Conv1 = 
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
        #self.AF1 = 
            nn.ReLU(inplace=True),
        #self.maxpool1 = 
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),     
        #self.AF2 =  
            nn.ReLU(inplace=True),
        #self.maxpool2 = 
            nn.MaxPool2d(kernel_size=3, stride=2),
        #self.Conv2 = 
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
        #self.AF3 = 
            nn.ReLU(inplace=True),
            #layer1 = nn.Sequential()

        #self.Conv3 = 
            nn.Conv2d(384, 256, kernel_size=3, padding=1),

        #self.features2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
                                       )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
               )
    def forward(self, x):
        x = self.features(x)
        conv5_features=x
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return conv5_features
#net = net.cuda()
transforms_ = [ transforms.Resize(int(260*1.12), Image.BICUBIC),
                transforms.RandomCrop((260, 640)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]

# the folder name is so long....
dataloader = DataLoader(ImageDataset("/home/timing/Git_Repos_Summer/PyTorch-GAN/implementations/cyclegan/test_results/Alderley_PyTorch_CycleGAN_epoch43_9thJan_trainA2testA", transforms_=transforms_, unaligned=True),batch_size = 1, shuffle=True, num_workers=1)
model_urls = {'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',}
def alexnet(pretrained = False, **kwargs):
    model = AlexNet(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
    return model
alexnet_model = alexnet()

alexnet_model.to(torch.device("cuda:2"))
# original saved file with DataParallel
state_dict = torch.load('/home/timing/.torch/models/alexnet-owt-4df8aa71.pth')
# create new OrderedDict that does not contain `module.`
#new_state_dict = OrderedDict()
#for k, v in state_dict.items():
#    name = k[7:] # remove `module.`
#    new_state_dict[name] = v
# load params
alexnet_model.load_state_dict(state_dict)

for i, batch in enumerate(dataloader):
    print(batch['A'].type)
    print(batch['A'].size)
    conv3_features = alexnet_model(batch['A'])
    feature_all.append(conv3_features.copy())
print(i)
feature_all = np.array(feature_all)
feature_all = feature_all.reshape(i,64896)
np.savetxt('temp_feature_all.txt',feature_all)

