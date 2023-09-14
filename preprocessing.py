import torch
import torch.nn as nn
import numpy as np
import cv2
from dataset import ckplus
from torchvision.models.detection import ssd
from torchvision import transforms
import dlib
from imutils import face_utils
from PIL import Image
from IPython.display import display
import h5py
import matplotlib.pyplot as plt


def normalise_image(x):
  x /= 255.0
  x -= 0.5
  x *= 2.0
  return x


class preprocessing_image:
    
    def __init__(self,clipLimit=2.0,tileGridSize=(8,8),display_annotated=False):
        self.clahe_object=cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
        self.detection_model=ssd.ssd300_vgg16(weights=ssd.SSD300_VGG16_Weights.DEFAULT)
        self.detection_model.eval()
        self.face_detector_landmark=dlib.get_frontal_face_detector()
        self.predictor_landmark=dlib.shape_predictor("/Users/yashsurange/Documents/GitHub/FER-from-static-images/shape_predictor_68_face_landmarks.dat")
        self.display_annotated=display_annotated
    
    def clahe(self,image):
        image=np.array(image)
        image = self.clahe_object.apply(image[:,:,0])     # cliplimit and brightness can be tuned
        image=np.stack((image,)*3,axis=-1)
        return image
    
    
    def face_detection(self,image):
        
        image=transforms.ToTensor()(image)
        
        with torch.no_grad():
            detections=self.detection_model([image])
        
        x, y, x_max, y_max = detections[0]['boxes'][0].tolist()     # A person will be detected here and so we have indexed using [0]
        face = transforms.ToPILImage()(image).crop((x, y, x_max, y_max))
        face= transforms.Resize((160, 160), interpolation=Image.BILINEAR)(face)
        return np.array(face)

    
    def visualize_face(self,im,landmarks):
        # Taken from https://github.com/rohan598/Landmark-Aware-Part-based-Ensemble-Transfer-Learning-Network-for-Facial-Expression-Recognition/blob/main/notebooks/summary_notebook.ipynb
        # draw box over face
        cv2.rectangle(im, (0,0), (160,160), (0,255,0), 2)
        for (x,y) in landmarks:
                cv2.circle(im, (x, y), 2, (0, 255, 0), -1)
        
        plt.imshow(im)
        plt.show()
            
        
    def landmark_annotation(self,image):
        # This detects the face in the image
        
        detect_face=self.face_detector_landmark(image,0)
    
        annotations=self.predictor_landmark(image,detect_face[0])
       
        annotations = face_utils.shape_to_np(annotations)
        
        if self.display_annotated==True:
            self.visualize_face(image,annotations)
        
        annotations=annotations.astype('float64').reshape((-1,))
        
        annotations = annotations/160
        return annotations



    
        
    
    
