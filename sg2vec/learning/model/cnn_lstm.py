import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class CNN_LSTM_Classifier(nn.Module):
    '''
    CNN+LSTM binary classifier
    To call module provide the input_shape and cfg params
    input_shape should be a tensor -> (batch_size, frames, channels, height, width) 
    '''
    def __init__(self, input_shape, cfg):
        super(CNN_LSTM_Classifier, self).__init__()
        self.cfg = cfg
        self.batch_size, self.frames, self.channels, self.height, self.width = input_shape
        self.dropout = nn.Dropout(self.cfg.training_configuration['dropout'])
        self.kernel_size = (3, 3)
        self.lstm_layers = 1
        self.conv_size = lambda i, k, p, s: int((i-k+2*p)/s + 1)
        self.pool_size = lambda i, k, p, s, pool : conv_size(i, k, p, s) // pool + 1
        self.flat_size = lambda f, h, w : f*h*w
        #if self.cfg.training_configuration['task_type'] == 'sequence_classification':
        self.TimeDistributed = lambda curr_layer, prev_layer : torch.stack([curr_layer(prev_layer[:,i]) for i in range(self.frames)], dim=1)
#        elif self.cfg.training_configuration['task_type'] == 'collision_prediction': #since evaluating frame by frame instead of as a whole sequence, we pass in 1
#          self.TimeDistributed = lambda curr_layer, prev_layer : torch.stack([curr_layer(prev_layer[:,i]) for i in range(1)], dim=1)

        # Note: conv_size and pool_size only work for square 2D matrices, if not a square matrix, run once for height dim and another time for width dim
        '''
        conv_size = lambda i, k, p, s: int((i-k+2*p)/s + 1)
        pool_size = lambda i, k, p, s, pool : conv_size(i, k, p, s) // pool + 1
        flat_size = lambda f, h, w : f*h*w
        '''
        #if self.cfg.training_configuration['task_type'] == 'sequence_classification':
        self.bn1 = nn.BatchNorm3d(num_features=5) #change from 5 to self.frame?
        self.bn2 = nn.BatchNorm3d(num_features=5)
        self.bn3 = nn.BatchNorm3d(num_features=5)
        self.bn4 = nn.BatchNorm1d(num_features=5)
        self.bn5 = nn.BatchNorm1d(num_features=5)
#        elif self.cfg.training_configuration['task_type'] == 'collision_prediction':
#          self.bn1 = nn.BatchNorm3d(num_features=1) #since evaluating frame by frame instead of as a whole sequence, we pass in 1
#          self.bn2 = nn.BatchNorm3d(num_features=1)
#          self.bn3 = nn.BatchNorm3d(num_features=1)
#          self.bn4 = nn.BatchNorm1d(num_features=1)
#          self.bn5 = nn.BatchNorm1d(num_features=1)       

        self.c1 = nn.Conv2d(in_channels=self.channels, out_channels=16, kernel_size=self.kernel_size)
        self.c2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=self.kernel_size)
        self.mp1 = nn.MaxPool2d(kernel_size=2)
        self.flat = nn.Flatten(start_dim=1)
        self.flat_dim = self.get_flat_dim()
        self.l1 = nn.Linear(in_features=self.flat_dim, out_features=200)
        self.l2 = nn.Linear(in_features=200, out_features=50)
        self.lstm1 = nn.LSTM(input_size=50, hidden_size=20, num_layers=self.lstm_layers, batch_first=True) 
        self.l3 = nn.Linear(in_features=20, out_features=2)
    
    def get_flat_dim(self):
        c1_h = self.conv_size(self.height, self.kernel_size[-1], 0, 1)
        c1_w = self.conv_size(self.width, self.kernel_size[-1], 0, 1)
        c2_h = self.conv_size(c1_h, self.kernel_size[-1], 0, 1)
        c2_w = self.conv_size(c1_w, self.kernel_size[-1], 0, 1)
        mp1_h = c2_h // 2
        mp1_w = c2_w // 2
        return self.flat_size(16, mp1_h, mp1_w)

    def forward(self, x):
        # Distribute learnable layers across all frames with shared weights
        if self.cfg.training_configuration['bnorm']: # can use a larger learning rate w/ bnorm #not in config currently
            
            c1 = F.relu(self.bn1(self.TimeDistributed(self.c1, x)))
            c2 = F.relu(self.bn2(self.TimeDistributed(self.c2, c1)))
            mp1 = self.dropout(self.bn3(self.TimeDistributed(self.mp1, c2)))
            flat = self.TimeDistributed(self.flat, mp1)
            l1 = F.relu(self.bn4(self.TimeDistributed(self.l1, flat)))
            l2 = F.relu(self.bn5(self.TimeDistributed(self.l2, l1)))
            if self.cfg.training_configuration["task_type"] == "collision_prediction":
              #import pdb;pdb.set_trace()
              l2 = l2.view(l2.shape[0]*l2.shape[1],50)
              l2 = l2.unsqueeze(1)
              
            _,(lstm1,_) = self.lstm1(l2)
            l3 = self.l3(lstm1)
        else:
            c1 = F.relu(self.TimeDistributed(self.c1, x))
            c2 = F.relu(self.TimeDistributed(self.c2, c1))
            mp1 = self.dropout(self.TimeDistributed(self.mp1, c2))
            flat = self.TimeDistributed(self.flat, mp1)
            l1 = F.relu(self.TimeDistributed(self.l1, flat))
            l2 = F.relu(self.TimeDistributed(self.l2, l1))
            if self.cfg.training_configuration["task_type"] == "collision_prediction":
              l2 = l2.view(l2.shape[0]*l2.shape[1],50)
              l2 = l2.unsqueeze(1)
            _,(lstm1,_) = self.lstm1(l2)
            l3 = self.l3(lstm1)
        
        self.layer_names = self.ordered_layers = [("c1", self.c1),("c2", self.c2),("mp1", self.mp1),("flat", self.flat), ("l1", self.l1),("l2", self.l2),("lstm1", self.lstm1),("l3", self.l3)]
        return l3.squeeze() 