import os
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np

'''
Dataset for landmarks detection model
'''

class landmark_dataset(Dataset):
    
    def transform(self, image):
        pass
        
    
    def __init__(self, normalised_images,annotations):
        self.normalised_images = normalised_images
        self.annotations = annotations
        
        
    def __len__(self):
        return len(self.normalised_images)
        
        

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img = Image.open(img_path).convert('L')  
        
        # Adding channels
        img=self.add_channels(img)
       
        # Transforming the image by resizing using bilinear interpolation
        img = self.transform(img)
            
        return img, label