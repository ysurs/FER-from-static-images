import torch
import torch.nn as nn
import numpy as np
import cv2
from dataset import ckplus
from torchvision.models.detection import ssd
from torchvision import transforms


class preprocessing_image:
    
    def clahe(self,image):
        image=np.array(image)
        # creating object to perform clahe
        clahe = cv2.createCLAHE(clipLimit=1)
        image = clahe.apply(image[:,:,0])+20     # cliplimit and brightness can be tuned
        image=np.stack((image,)*3,axis=-1)
        return image
    
    
    def face_detection(self,image):
        # Get the pretrained model on COCO dataset
        detection_model=ssd.ssd300_vgg16(weights=ssd.SSD300_VGG16_Weights.DEFAULT)
        image=transforms.ToTensor()(image)
        
        detection_model.eval()
        with torch.no_grad():
            detections=detection_model([image])
        
        x, y, x_max, y_max = detections[0]['boxes'][0].tolist()
        face = transforms.ToPILImage()(image).crop((x, y, x_max, y_max))
        face=torch.tensor(np.array(face))
        return face
        
        
    def landmark_annotation(self,image):
        # To use dlib, have figured out in check.ipynb
        pass

    
    
def main():
    data=ckplus('./CK+_Complete/')
    
    
    

if __name__ == '__main__':
    main()