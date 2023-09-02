import torch
import torch.nn as nn
from components import expression_classification, feature_extractor
from preprocessing import preprocessing_image


'''
1. Consists of feature extractor and classification head
2. Annotation is not needed here
3. input should be image after augmentation, label
4. An ensemble of five randomly initialized Baseline networks is trained for expression prediction.
'''

class baseline(nn.Module):
    
    def __init__(self,in_channel,out_channel,no_of_expressions):
        
        super(baseline, self).__init__()
        
        self.preprocessing_object=preprocessing_image()
        self.baseline_feature_extractor=feature_extractor(in_channel, out_channel)
        self.baseline_expression_classification=expression_classification(no_of_expressions)
        
    
    def forward(self,image_batch):
        image_after_clahe=self.preprocessing_object.clahe(image_batch)
        face_detection=self.preprocessing_object.face_detection(image_after_clahe)
        # feature_vector=self.baseline_feature_extractor(face_detection)
        # expression=self.baseline_expression_classification(feature_vector)
        return face_detection
        
    
    