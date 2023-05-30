import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, SubsetRandomSampler

import torch
import torchvision
from torchvision import transforms

from load_data import *
import os
import json
from PIL import Image
from torch.utils.data import Dataset

import torch
import torchvision
import torchvision.io as torch_io
import torchvision.transforms as transforms

# image and annotation directories
image_dir = 'data/train/images'
annotation_dir = 'data/train/annotations'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = {'dot':1, 'vertical_bar':2, 'line':3, 'scatter':4, 'horizontal_bar':5}

class graphdata(Dataset):
    def __init__(self, image_dir, annotation_dir, transform=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        self.image_files = [file for file in os.listdir(image_dir) if file.endswith('.jpg')]
        # print(self.image_files)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image_path = os.path.join(self.image_dir, image_file)
        annotation_file = image_file.replace('.jpg', '.json')
        annotation_path = os.path.join(self.annotation_dir, annotation_file)
        # print(image_path)
        # print(annotation_path)

        # load image
        image = torch_io.read_image(image_path).type(torch.FloatTensor)
        
        with open(annotation_path, 'r') as f:
            annotation = json.load(f)

        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(classes[annotation['chart-type']])

# image transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


dataset = graphdata(image_dir, annotation_dir, transform=transform)
index = list(range(len(dataset)))

#dataset distributions
train = .7
test = .2
validation = .1

#generate random indexes into the dataset for each dataloader
np.random.shuffle(index)
split = int(np.floor(train * len(dataset)))
train_indices, remaining = index[split:], index[:split]
split = int(np.floor(.67 * len(dataset)))
test_indices, validation_indices = remaining[split:], remaining[:split]

train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)
validation_sampler = SubsetRandomSampler(validation_indices)

#create a dataloader for training, testing, and validation
train_data_loader = DataLoader(dataset, batch_size=16, sampler=train_sampler)
test_data_loader = DataLoader(dataset, batch_size=16, sampler=test_sampler)
validation_data_loader = DataLoader(dataset, batch_size=16, sampler=validation_sampler)


