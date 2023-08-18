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
            self.convolutional_block(64,128),
            nn.MaxPool2d(2,2)
            
        )
        
        
        
    
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
        pass
    
    
