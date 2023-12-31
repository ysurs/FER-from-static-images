import torch
import torch.nn as nn


# class and functions for feature extraction
class feature_extractor(nn.Module):
    
    def __init__(self,fe_in_channels,fe_out_channels):
        super(feature_extractor, self).__init__()
        
        self.block_sequence=nn.Sequential(
            self.convolutional_block(fe_in_channels,16),
            nn.MaxPool2d(2,2),
            self.convolutional_block(16,32),
            nn.MaxPool2d(2,2),
            self.convolutional_block(32,64),
            nn.MaxPool2d(2,2),
            self.convolutional_block(64,fe_out_channels),
            nn.MaxPool2d(2,2)
        )
        self.global_average_pooling=nn.AvgPool2d(kernel_size=10)
        self.initialize_weights()
        
    
    
    def initialize_weights(self):
        for i in range(len(self.block_sequence)):
            if i%2==0:
                for layer in self.block_sequence[i]:
                    if isinstance(layer,nn.Conv2d):
                        nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
                        
        
        
        
        
    
    def convolutional_block(self,in_channels,out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=(3,3),padding="same"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels,out_channels,kernel_size=(3,3),padding="same"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
    
    def forward(self,input):
        extracted_features=self.block_sequence(input)
        feature_vector=self.global_average_pooling(extracted_features)
        no_of_images=feature_vector.shape[0]    # extracting no of images in batch
        feature_vector=feature_vector.view(no_of_images,-1)
        return feature_vector



# Class for creating the header layer for landmark localisation
class landmark_localization(nn.Module):
    
    def __init__(self,no_of_landmarks):
        
        super(landmark_localization,self).__init__()
        
        self.dense_layers=nn.Sequential(
            nn.Linear(128,128),
            nn.Linear(128,no_of_landmarks)
        )
        self.landmark_weight_initialisation()
        
        
    def landmark_weight_initialisation(self):
        
        for layer in self.dense_layers:
            nn.init.xavier_uniform_(layer.weight)
            
                
    
    def forward(self,vector):
        return self.dense_layers(vector)
    


# class for expression classification
class expression_classification(nn.Module):
    
    def __init__(self,no_of_expressions):
        
        super(expression_classification,self).__init__()
        
        self.layers=nn.Sequential(
            nn.Linear(128,128),
            nn.Linear(128,128),
            nn.Linear(128,no_of_expressions),
            nn.Softmax(dim=1)
        )
        self.expression_classification_initialization()
        
        
    def expression_classification_initialization(self):
        
        for layer in self.layers:
            if isinstance(layer,nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
                
    
    def forward(self,vector):
        return self.layers(vector)
    
    



class landmark_model(nn.Module):
    def __init__(self,feature_extractor_in_channels, feature_extractor_out_channels, no_of_landmarks):
        super(landmark_model,self).__init__()
        self.feature_extractor_landmark=feature_extractor(feature_extractor_in_channels,feature_extractor_out_channels)
        self.landmark_detection=landmark_localization(no_of_landmarks)
        
    def forward(self,input):
        return self.landmark_detection(self.feature_extractor_landmark(input))
    


class expression_model(nn.Module):
    def __init__(self,feature_extractor_in_channels, feature_extractor_out_channels, no_of_expressions):
        super(expression_model,self).__init__()
        self.feature_extractor_expression=feature_extractor(feature_extractor_in_channels,feature_extractor_out_channels)
        self.expression_recognizer=expression_classification(no_of_expressions)
        
    def forward(self,input):
        return self.expression_recognizer(self.feature_extractor_expression(input))