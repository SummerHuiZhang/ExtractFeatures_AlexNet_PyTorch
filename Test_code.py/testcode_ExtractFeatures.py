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

__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            #layer1 = nn.Sequential()
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            )
            #######################################
            ####delete the layer after conv3 ######
            #######################################
            #layer1.add_module('conv3' = nn.Conv2d(384, 256), kernel_size=3, padding=1)
            
            #nn.ReLU(inplace=True),
            #nn.Conv2d(256, 256, kernel_size=3, padding=1),
            #nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=3, stride=2),
        #)

        #self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        #self.classifier = nn.Sequential(
        #    nn.Dropout(),
        #    nn.Linear(256 * 6 * 6, 4096),
        #    nn.ReLU(inplace=True),
        #    nn.Dropout(),
        #    nn.Linear(4096, 4096),
        #    nn.ReLU(inplace=True),
        #    nn.Linear(4096, num_classes),
        #)

    def forward(self, x):
        x = self.features(x)
        #x = self.avgpool(x)
        #self.feature = x
        x = x.view(x.size(0), 256 * 6 * 6)
        #x = self.classifier(x)
        return x
net = AlexNet()
alexnet_model=torchvision.models.alexnet(pretrained=True)

#net = net.cuda()
transforms_ = [ transforms.Resize(int(260*1.12), Image.BICUBIC),
                transforms.RandomCrop((260, 640)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]

dataloader = DataLoader(ImageDataset("/home/timing/Git_Repos_Summer/PyTorch-GAN/implementations/cyclegan/test_results/Alderley_PyTorch_CycleGAN_epoch43_9thJan_trainA2testA", transforms_=transforms_, unaligned=True),batch_size = 1, shuffle=True, num_workers=1)

for i, batch in enumerate(dataloader):
    # load images with Dataloader
    #real_A = Variable(batch['A'].type(Tensor))
    convs_features = alexnet_model(batch)
    feature_all.append(conv3_features.copy())
conv3_features = conv3_features.reshape(9,64896)
print(i)
feature_all = np.array(feature_all)
feature_all = feature_all.reshape(i,64896)#mean = np.loadtxt('mean_0501.txt')
#component = np.loadtxt('comonents_0501.txt')
np.savetxt('temp_feature_all.txt',feature_all)
