import os
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class ckplus(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
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
    
    def transform(self, image):
        pass

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img = Image.open(img_path).convert('L')  
        
        # if self.transform:
        #     img = self.transform(img)
            
        return img, label