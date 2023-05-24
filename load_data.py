import os
import json
from PIL import Image
from torch.utils.data import Dataset


import torch
import torchvision
import torchvision.io as torch_io
import torchvision.transforms as transforms

class graphdata(Dataset):
    def __init__(self, image_dir, annotation_dir, transform=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        self.image_files = [file for file in os.listdir(image_dir) if file.endswith('.jpg')]
        #print(self.image_files)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image_path = os.path.join(self.image_dir, image_file)
        annotation_file = image_file.replace('.jpg', '.json')
        annotation_path = os.path.join(self.annotation_dir, annotation_file)
        #print(image_path)
        #print(annotation_path)

        # Load the image
        image = Image.open(image_path).convert("RGB")

        #image = torch_io.read_image(image_path).type(torch.FloatTensor)

        # Load the annotation
        with open(annotation_path, 'r') as f:
            annotation = json.load(f)

        # Apply transformations if provided
        if self.transform is not None:
            image = self.transform(image)
            transform = transforms.Compose([transforms.ToTensor()])
            image = transform(image)
        #image = image.float()


        return image, annotation

