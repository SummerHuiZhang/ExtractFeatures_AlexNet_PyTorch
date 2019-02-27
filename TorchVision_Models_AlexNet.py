import sys
sys.path.append('..')
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import cv2
import numpy as np
from torch.autograd import Variable
from PIL import Image
from sklearn.decomposition import IncrementalPCA
import matplotlib.pyplot as plt
import glob
import torchvision.transforms as transforms


image_name_list_file = sys.argv[1]
image_name_list = open(image_name_list_file)
image_number = int(sys.argv[2]) 

__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        #self.features = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Conv2d(64, 192, kernel_size=5, padding=2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Conv2d(192, 384, kernel_size=3, padding=1),        
        nn.ReLU(inplace=True),
        self.conv3 = nn.Conv2d(384, 256, kernel_size=3, padding=1),

        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),
        #)
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
        #x = self.features(x)
        #x = self.avgpool(x)
        #x = x.view(x.size(0), 256 * 6 * 6)
        x = self.conv3(x)
        return x
net = AlexNet()
#print(net)
print(image_number/9)
for i in range(0,int(image_number/9)):
    img_batch=[]
    for j in range(0,9):   #includes 0-9
        image_name = image_name_list.readline()[:-1]
        #im = Image.open(image_name)
        im = Image.open(image_name).convert('RGB')
        trans = transforms.ToPILImage()
        trans1 = transforms.ToTensor()
        #im = DataLoader(ImageDataset("/bak/datasets/%s" % opt.dataset_name, transforms_=transforms_, unaligned=True)

        # Set model input
        #im = Variable(im.type(Tensor))
        im = trans1(im)
        #in_ = np.array(im, dtype=np.float32)
        #in_ = in_[:,:,::-1]#change image channel to BGR
        #in_ = cv2.resize(in_,(227,227))
        #in_ -= mean
        #in_ = in_.transpose((2,0,1))
        #img_batch.append(in_.copy())
        # load net
        # shape for input (data blob is N x C x H x W), set data

        #out=net(im
        #print(out)
        convs_features = net(im)
        print('features of conv3',conv3_features)
        conv3_features = conv3_features.reshape(9,64896)
        feature_all.append(conv3_features.copy())
        print(i)
feature_all = np.array(feature_all)
feature_all = feature_all.reshape(image_numbeer,64896)#mean = np.loadtxt('mean_0501.txt')
#component = np.loadtxt('comonents_0501.txt')
#mean = np.matrix(mean)
#component = np.matrix(component)
#feature_all = np.matrix(feature_all)
#feature_316 = component*(feature_all.T-mean.T)
#feature_316 = feature_316.T
np.savetxt('Features_Alderley_Epoch43_FakedTrainA_4788Images.txt',feature_all)

