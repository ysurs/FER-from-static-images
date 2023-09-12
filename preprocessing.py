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
        try:
            annotations=self.predictor_landmark(image,detect_face[0])
        except:
            print("No")
            return
        annotations = face_utils.shape_to_np(annotations)
        
        if self.display_annotated==True:
            self.visualize_face(image,annotations)
        
        annotations = annotations/160
        return annotations


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
    
    images_after_clahe=np.array(images_after_clahe,dtype=np.uint8)
    
    # Getting face crops from all the images
    face_crops=[]
    
    for i in range(images_after_clahe.shape[0]):
        face_crops.append(preprocessing_object.face_detection(images_after_clahe[i]))
    
    # Removing images for which detection is not working : index: 1576 and 1630
    face_crops.pop(1576)
    face_crops.pop(1630)
    face_crops.pop(1629)
    
    image_landmarks=[]
    
    for i in range(len(face_crops)):
        image_landmarks.append(preprocessing_object.landmark_annotation(face_crops[i]))
        
    face_crops=np.array(face_crops).astype('float64')
    
    normalised_images=normalise_image(face_crops)
    
    print(normalised_images.shape)
    print(len(image_landmarks))

    
    # # Storing numpy arrays for image and labels in using h5py
    # with h5py.File('images_after_clahe.h5', 'w') as file:
    #      dataset = file.create_dataset('after_clahe', data=images_after_clahe)
    
    # with h5py.File('labels.h5', 'w') as file:
    #      dataset = file.create_dataset('image_labels', data=total_ckplus_labels)
    
        
    
    
