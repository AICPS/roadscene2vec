import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class CNN_Classifier(nn.Module):
    '''
    3D CNN+Linear binary classifier
    To call module provide the input_shape and cfg params
    input_shape should be a tensor -> (batch_size, frames, channels, height, width) 
    '''
    def __init__(self, input_shape, cfg):
        super(CNN_Classifier, self).__init__()
        self.cfg = cfg
        self.batch_size, self.frames, self.channels, self.height, self.width = input_shape
        self.kernel_size = (1, 5, 5)
        self.conv_size = lambda i, k, p, s: int((i-k+2*p)/s + 1)
        self.pool_size = lambda i, k, p, s, pool : conv_size(i, k, p, s) // pool + 1

        self.c1 = nn.Conv3d(in_channels=self.channels, out_channels=32, kernel_size=self.kernel_size)
        self.c2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=self.kernel_size)
        self.mp1 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        self.mp2 = nn.MaxPool3d(kernel_size=(1,2,2))
        self.flat = nn.Flatten(start_dim=1)
        self.flat_dim = 64*self.frames*self.get_flat_dim() # TODO: automate this number
        self.l1 = nn.Linear(in_features=self.flat_dim, out_features=1000)
        self.l2 = nn.Linear(in_features=1000, out_features=2)
    
    def get_flat_dim(self):
        c1_h = self.conv_size(self.height, self.kernel_size[-1], 0, 1)
        c1_w = self.conv_size(self.width, self.kernel_size[-1], 0, 1)
        mp1_h = c1_h // 2
        mp1_w = c1_w // 2
        c2_h = self.conv_size(mp1_h, self.kernel_size[-1], 0, 1)
        c2_w = self.conv_size(mp1_w, self.kernel_size[-1], 0, 1)
        mp2_h = c2_h // 2
        mp2_w = c2_w // 2
        return mp2_h * mp2_w

    def reshape(self, x):
        # assumes batch first dim
        return x.permute(0, 2, 1, 3, 4)

    def forward(self, x):
        # format input for 3d cnn
        assert len(x.shape) == 5
        x = self.reshape(x)
        c1 = F.relu(self.c1(x))
        mp1 = self.mp1(c1)
        c2 = F.relu(self.c2(mp1))
        mp2 = self.mp2(c2)
        flat1 = self.flat(mp2)
        l1 = F.relu(self.l1(flat1))
        l2 = self.l2(l1)
        return l2.squeeze()