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
        return np.array(face)
        
        
    def landmark_annotation(self,image):
        # The image passed is a numpy array
        # To use dlib, have figured out in check.ipynb
        pretrained_model="shape_predictor_68_face_landmarks.dat"
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(pretrained_model)
        
        # This detects the face in the image
        detect_face=detector(image,0)
        
        # At present, this we are considering only 1 image, will add support for images in batches
        annotations=predictor(image,detect_face[0])
        annotations = face_utils.shape_to_np(annotations)
        
        for (x,y) in annotations:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
            
        image_pil = Image.fromarray(image)
        display(image_pil)
        return 

