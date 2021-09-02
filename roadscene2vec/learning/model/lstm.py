import torch
import torch.nn as nn


class LSTM_Classifier(nn.Module):
    '''
    Recurrent Network binary classifier
    Supports 3 models: {GRU, LSTM, LSTM+Dropout}
    
    To call module provide the input_shape, model_name, and cfg params
    input_shape should be a tensor -> (batch_size, frames, channels, height, width) 
    model_name must be one of these {gru, lstm}
        the lstm model can be configured with dropout if cfg.dropout > 0
    '''
    def __init__(self, input_shape, model_name, cfg):
        super(LSTM_Classifier, self).__init__()
        self.cfg = cfg
        self.model_name = model_name
        self.batch_size, self.frames, self.channels, self.height, self.width = input_shape
        self.dropout = nn.Dropout(self.cfg.model_configuration['dropout'])

        if self.model_name == 'gru':
            self.l1 = nn.GRU(input_size=self.channels*self.height*self.width, hidden_size=100, batch_first=True)
            self.l2 = nn.Linear(in_features=100, out_features=2)
        
        elif self.model_name == 'lstm':
            self.l1 = nn.LSTM(input_size=self.channels*self.height*self.width, hidden_size=512, batch_first=True)
            self.l2 = nn.LSTM(input_size=512, hidden_size=512, batch_first=True)
            self.l3 = nn.Linear(in_features=512, out_features=1000)
            self.l4 = nn.Linear(in_features=1000, out_features=200)
            self.l5 = nn.Linear(in_features=200, out_features=2)

    def forward(self, x):
        # format input for lstm
        # x = self.reshape(x)
        x = torch.reshape(x, (x.shape[0], x.shape[1], -1))
        if self.model_name == 'gru':
            if self.cfg.training_configuration["task_type"] == "collision_prediction":
              l1,_ = self.l1(x) 
              l1 = l1.reshape(l1.shape[0]*l1.shape[1],100)
              l1 = l1.unsqueeze(1)
            elif self.cfg.training_configuration["task_type"] == "sequence_classification":
              _,l1 = self.l1(x) # return only last sequence
            l2 = self.l2(l1)
            return l2.squeeze()
        elif self.model_name == 'lstm':
            dropout = lambda curr_layer: self.dropout(curr_layer) if self.cfg.model_configuration['dropout'] != 0 else curr_layer
            l1,_ = self.l1(x)  # return all sequences
            if self.cfg.training_configuration["task_type"] == "collision_prediction":
              l1 = l1.reshape(l1.shape[0]*l1.shape[1],512)
              l1 = l1.unsqueeze(1)
            _,(l2,_) = self.l2(l1) # return only last sequence
            l3 = self.l3(dropout(l2))
            l4 = self.l4(dropout(l3))
            l5 = self.l5(l4)
            return l5.squeeze()
        else:
            raise Exception('Unsupported model! Choose between gru or lstm') 