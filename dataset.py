import os
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np


class ckplus(Dataset):
    
    def transform(self, image):
        resize_transform = transforms.Compose([
        transforms.ToPILImage(),  # Convert to PIL image
        transforms.Resize((300, 300), interpolation=Image.BILINEAR)  # Resize to 300x300
        ])
        image=np.array(resize_transform(image))
        return image
        
    
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        self.data = []
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_name)
                self.data.append((img_path, self.class_to_idx[cls_name]))
        
    def __len__(self):
        return len(self.data)
        
    # To make a gray scaled image into a 3 channeled image
    def add_channels(self,image):
        image=np.array(image)
        image=np.stack((image,)*3,axis=-1)
        return image
        

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img = Image.open(img_path).convert('L')  
        
        # Adding channels
        img=self.add_channels(img)
       
        # Transforming the image by resizing using bilinear interpolation
        img = self.transform(img)
            
        return img, label