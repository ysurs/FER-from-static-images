import torch
import torch.nn as nn



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
                        nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                        if layer.bias is not None:
                            nn.init.constant_(layer.bias, 0)
        
        return
        
        
    
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
        feature_vector=self.global_average_pooling(extracted_features).reshape(1,-1)
        return feature_vector
    
    
