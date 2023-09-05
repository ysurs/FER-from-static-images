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

class preprocessing_image:
    
    def clahe(self,image):
        image=np.array(image)
        # creating object to perform clahe
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        image = clahe.apply(image[:,:,0])     # cliplimit and brightness can be tuned
        image=np.stack((image,)*3,axis=-1)
        return image
    
    
    def face_detection(self,image):
        # Get the pretrained model on COCO dataset- This impelementation is very slow- needs change
        detection_model=ssd.ssd300_vgg16(weights=ssd.SSD300_VGG16_Weights.DEFAULT)
        image=transforms.ToTensor()(image)
        
        detection_model.eval()
        with torch.no_grad():
            detections=detection_model([image])
        
        x, y, x_max, y_max = detections[0]['boxes'][0].tolist()     # A person will be detected here and so we have indexed using [0]
        face = transforms.ToPILImage()(image).crop((x, y, x_max, y_max))
        face= transforms.Resize((160, 160), interpolation=Image.BILINEAR)(face)
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
        
        
        ### Use below logic for actual face annotation
        for (x,y) in annotations:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
            
        return image


if __name__ == "__main__":

    # Creating an instance of the class for ckplus dataset
    ckplus_dataset=ckplus('./CK+_Complete/')
    
    total_ckplus_images=[]
    total_ckplus_labels=[]

    # Getting total images and labels
    for i in range(ckplus_dataset.__len__()):
        total_ckplus_images.append(ckplus_dataset.__getitem__(i)[0])
        total_ckplus_labels.append(ckplus_dataset.__getitem__(i)[1])
    
    # Changing into numpy arrays
    total_ckplus_images=np.array(total_ckplus_images)
    total_ckplus_labels=np.array(total_ckplus_labels)
    
    
    # Object for preprocessing- everything from clahe, face detection
    preprocessing_object=preprocessing_image()
    
    # Applying clahe to all the images
    images_after_clahe=[]
    
    for i in range(total_ckplus_images.shape[0]):
        images_after_clahe.append(preprocessing_object.clahe(total_ckplus_images[i]))
    
    images_after_clahe=np.array(images_after_clahe)
    
    
    # Storing numpy arrays for image and labels in using h5py
    with h5py.File('images_after_clahe.h5', 'w') as file:
         dataset = file.create_dataset('after_clahe', data=images_after_clahe)
    
    with h5py.File('labels.h5', 'w') as file:
         dataset = file.create_dataset('image_labels', data=total_ckplus_labels)
    
        
    
    
