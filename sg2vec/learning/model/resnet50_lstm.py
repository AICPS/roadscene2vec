import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResNet50_LSTM_Classifier(nn.Module):
    '''
    ResNet50+LSTM binary classifier
    
    To call module provide the input_shape, model_name, and cfg params
    input_shape should be a tensor -> (batch_size, frames, channels, height, width) 
    '''
    def __init__(self, input_shape, cfg):
        super(ResNet50_LSTM_Classifier, self).__init__()
        self.cfg = cfg
        self.batch_size, self.frames, self.channels, self.height, self.width = input_shape
  
        # “Deep Residual Learning for Image Recognition”
        # Using only feature extraction layers shape: (C, H, W) -> (2048, 1, 1)
        '''
        self.resent = models.resnet50(pretrained=True, progress=True)
        nn.Sequential(*list(self.resnet.children())[:-3])(x[0]).shape
        torch.Size([16, 512, 28, 28])
        '''
        self.resnet = nn.Sequential(*list(models.resnet50(pretrained=True, progress=True).children())[:-1])
        
        # TODO: verify lstm hidden size with louis
        # self.lstm1 = nn.LSTM(input_size=512, hidden_size=20)
        self.lstm1 = nn.LSTM(input_size=2048, hidden_size=20, batch_first=True)
        self.l1 = nn.Linear(in_features=20, out_features=2)

    def forward(self, x):
        TimeDistributed = lambda curr_layer, prev_layer : torch.stack([curr_layer(prev_layer[:,i]) for i in range(self.frames)], dim=1)
        resnet = TimeDistributed(self.resnet, x)
        if self.cfg.training_configuration["task_type"] == "collision_prediction": 
          lstm1,_ = self.lstm1(torch.squeeze(resnet))
          lstm1 = lstm1.reshape(lstm1.shape[0]*lstm1.shape[1],20)
        elif self.cfg.training_configuration["task_type"] == "sequence_classification":
          _,(lstm1,_) = self.lstm1(torch.squeeze(resnet))
        l1 = self.l1(lstm1)
        return l1.squeeze()