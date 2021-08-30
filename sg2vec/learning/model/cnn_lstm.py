import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.dropout = self.cfg.model_configuration['dropout']
        self.kernel_size = (3, 3)
        self.lstm_layers = 1
        self.conv_size = lambda i, k, p, s: int((i-k+2*p)/s + 1)
        self.pool_size = lambda i, k, p, s, pool : self.conv_size(i, k, p, s) // pool + 1
        self.flat_size = lambda f, h, w : f*h*w
        self.TimeDistributed = lambda curr_layer, prev_layer : torch.stack([curr_layer(prev_layer[:,i]) for i in range(self.frames)], dim=1)
        self.enable_bnorm = self.cfg.model_configuration['bnorm']
        # Note: conv_size and pool_size only work for square 2D matrices, if not a square matrix, run once for height dim and another time for width dim
        '''
        conv_size = lambda i, k, p, s: int((i-k+2*p)/s + 1)
        pool_size = lambda i, k, p, s, pool : conv_size(i, k, p, s) // pool + 1
        flat_size = lambda f, h, w : f*h*w
        '''
        self.bn1 = nn.BatchNorm3d(num_features=self.frames)
        self.bn2 = nn.BatchNorm3d(num_features=self.frames)
        self.bn3 = nn.BatchNorm3d(num_features=self.frames)
        self.bn4 = nn.BatchNorm1d(num_features=self.frames)
        self.bn5 = nn.BatchNorm1d(num_features=self.frames)    

        self.c1 = nn.Conv2d(in_channels=self.channels, out_channels=16, kernel_size=self.kernel_size)
        self.c2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=self.kernel_size)
        self.mp1 = nn.MaxPool2d(kernel_size=2)
        self.flat = nn.Flatten(start_dim=1)
        self.flat_dim = self.get_flat_dim()
        self.l1 = nn.Linear(in_features=self.flat_dim, out_features=200)
        self.l2 = nn.Linear(in_features=200, out_features=50)
        self.lstm1 = nn.LSTM(input_size=50, hidden_size=20, num_layers=self.lstm_layers, batch_first=True) 
        self.l3 = nn.Linear(in_features=20, out_features=2)
        self.layer_names = self.ordered_layers = [("c1", self.c1),("c2", self.c2),("mp1", self.mp1),("flat", self.flat), ("l1", self.l1),("l2", self.l2),("lstm1", self.lstm1),("l3", self.l3)]

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
        if self.enable_bnorm: # can use a larger learning rate w/ bnorm #not in config currently
            x = F.relu(self.bn1(self.TimeDistributed(self.c1, x)))
            x = F.relu(self.bn2(self.TimeDistributed(self.c2, x)))
            x = F.dropout(self.bn3(self.TimeDistributed(self.mp1, x)), p=self.dropout, training=self.training)
            x = F.dropout(self.TimeDistributed(self.flat, x), p=self.dropout, training=self.training)
            x = F.dropout(F.relu(self.bn4(self.TimeDistributed(self.l1, x))), p=self.dropout, training=self.training)
            x = F.dropout(F.relu(self.bn5(self.TimeDistributed(self.l2, x))), p=self.dropout, training=self.training)
        else:
            x = F.relu(self.TimeDistributed(self.c1, x))
            x = F.relu(self.TimeDistributed(self.c2, x))
            x = F.dropout(self.TimeDistributed(self.mp1, x), p=self.dropout, training=self.training)
            x = F.dropout(self.TimeDistributed(self.flat, x), p=self.dropout, training=self.training)
            x = F.dropout(F.relu(self.TimeDistributed(self.l1, x)), p=self.dropout, training=self.training)
            x = F.dropout(F.relu(self.TimeDistributed(self.l2, x)), p=self.dropout, training=self.training)
        if self.cfg.training_configuration["task_type"] == "collision_prediction":
            x = x.view(x.shape[0]*x.shape[1],50)
            x = x.unsqueeze(1)
        _,(x,_) = self.lstm1(x)
        x = torch.squeeze(self.l3(x))
        return F.log_softmax(x, dim=-1)