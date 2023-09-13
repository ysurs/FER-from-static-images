"""
Trains a PyTorch model
"""
import os
import torch
from torchvision import transforms
from dataset import ckplus
from preprocessing import preprocessing_image,normalise_image
import numpy as np


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
    
    # Removing images for which detection is not working : index: 1576 ,1630,1629
    face_crops.pop(1576)
    face_crops.pop(1630)
    face_crops.pop(1629)
    
    image_landmarks=[]
    
    for i in range(len(face_crops)):
        image_landmarks.append(preprocessing_object.landmark_annotation(face_crops[i]))
        
    face_crops=np.array(face_crops).astype('float64')
    
    normalised_images=normalise_image(face_crops)
    image_landmarks=np.array(image_landmarks)
    
    '''
        1. split the dataset into train and validation
        2. create custom dataset for training and validation
        3. create training script
        4  step 3 and 4 to be done parallely
    '''


    

# # Setup directories
# train_dir = "data/pizza_steak_sushi/train"
# test_dir = "data/pizza_steak_sushi/test"

# # Setup target device
# device = "cuda" if torch.cuda.is_available() else "cpu"

# # Create transforms
# data_transform = transforms.Compose([
#   transforms.Resize((64, 64)),
#   transforms.ToTensor()
# ])

# # Create DataLoaders with help from data_setup.py
# train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
#     train_dir=train_dir,
#     test_dir=test_dir,
#     transform=data_transform,
#     batch_size=BATCH_SIZE
# )

# # Create model with help from model_builder.py
# model = model_builder.TinyVGG(
#     input_shape=3,
#     hidden_units=HIDDEN_UNITS,
#     output_shape=len(class_names)
# ).to(device)

# # Set loss and optimizer
# loss_fn = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(),
#                              lr=LEARNING_RATE)

# # Start training with help from engine.py
# engine.train(model=model,
#              train_dataloader=train_dataloader,
#              test_dataloader=test_dataloader,
#              loss_fn=loss_fn,
#              optimizer=optimizer,
#              epochs=NUM_EPOCHS,
#              device=device)

# # Save the model with help from utils.py
# utils.save_model(model=model,
#                  target_dir="models",
#                  model_name="05_going_modular_script_mode_tinyvgg_model.pth")