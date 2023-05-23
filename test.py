import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision
from load_data import *
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

#image and annotation directories
image_dir = 'data/train/images'
annotation_dir = 'data/train/annotations'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""

"""
#image transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    
])
def collate_fn(batch):
    images = []
    annotations = []

    for image, annotation in batch:
        images.append(image)
        annotations.append(annotation)

    return images, annotations



dataset = graphdata(image_dir, annotation_dir, transform=transform)

# Create a data loader
data_loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)


for batch_idx, (images, annotations) in enumerate(data_loader):
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

# Iterate over the data loader in your training loop
