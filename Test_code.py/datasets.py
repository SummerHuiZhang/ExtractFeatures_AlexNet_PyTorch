import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.files_A = sorted(glob.glob(root + '/*.*'))
    def __getitem__(self, index):
        A_item = self.transform(Image.open(self.files_A[index % len(self.files_A)]))
        return {'A': A_item}
    def __len__(self):
        return len(self.files_A)
                                                                
