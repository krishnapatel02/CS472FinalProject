import torch
from torchvision import transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision
from load_data import *
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

#image and annotation directories
image_dir = 'data/train/images'
annotation_dir = 'data/train/annotations'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train = .7
test = .2
validation = .1

"""

"""
#image transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    
])
def collate_fn(batch): #makes the input to the tensors the same size 
    images = []
    annotations = []

    for image, annotation in batch:
        images.append(image)
        annotations.append(annotation)

    return images, annotations



dataset = graphdata(image_dir, annotation_dir, transform=transform)
index = list(range(len(dataset)))
np.random.shuffle(index)
split = int(np.floor(train * len(dataset)))
train_indices, remaining = index[split:], index[:split]
split = int(np.floor(.67 * len(dataset)))
test_indices, validation_indices = remaining[split:], remaining[:split]

train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)
validation_sampler = SubsetRandomSampler(validation_indices)





# Create a data loader
train_data_loader = DataLoader(dataset, batch_size=16,  collate_fn=collate_fn, sampler=train_sampler)
test_data_loader = DataLoader(dataset, batch_size=16,  collate_fn=collate_fn, sampler=test_sampler)
validation_data_loader = DataLoader(dataset, batch_size=16, collate_fn=collate_fn, sampler=validation_sampler)

"""
for batch_idx, (images, annotations) in enumerate(validation_data_loader):
    # Check the loaded images
    for image in images:
        # Display or process the image as needed
        t = transforms.ToPILImage()
        img = t(image)  # Show the image using PIL.Image
        # Or convert image to ndarray and visualize using matplotlib, for example
        img.show()
        print(annotations)
    # Break the loop after inspecting a few batches
    if batch_idx >= 2:
        break

"""