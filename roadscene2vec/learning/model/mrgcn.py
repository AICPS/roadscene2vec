import os
import sys

sys.path.append(os.path.dirname(sys.path[0]))
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchnlp.nn import Attention
from torch.nn import Linear, LSTM
from torch_geometric.nn import RGCNConv, TopKPooling, FastRGCNConv
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from .rgcn_sag_pooling import RGCNSAGPooling

class MRGCN(nn.Module):
    
    def __init__(self, config):
        super(MRGCN, self).__init__()
        self.num_features = config.model_configuration['num_of_classes']
        self.num_relations = config.model_configuration['num_relations']
        self.num_classes  = config.model_configuration['nclass']
        self.num_layers = config.model_configuration['num_layers'] #defines number of RGCN conv layers.
        self.hidden_dim = config.model_configuration['hidden_dim']
        self.layer_spec = None if config.model_configuration['layer_spec'] == None else list(map(int, config.model_configuration['layer_spec'].split(',')))
        self.lstm_dim1 = config.model_configuration['lstm_input_dim']
        self.lstm_dim2 = config.model_configuration['lstm_output_dim']
        self.rgcn_func = FastRGCNConv if config.model_configuration['conv_type'] == "FastRGCNConv" else RGCNConv
        self.activation = F.relu if config.model_configuration['activation'] == 'relu' else F.leaky_relu
        self.pooling_type = config.model_configuration['pooling_type']
        self.readout_type = config.model_configuration['readout_type']
        self.temporal_type = config.model_configuration['temporal_type']

        self.dropout = config.model_configuration['dropout']
        self.conv = []
        total_dim = 0

        if self.layer_spec == None:
            if self.num_layers > 0:
                self.conv.append(self.rgcn_func(self.num_features, self.hidden_dim, self.num_relations).to(config.model_configuration['device']))
                total_dim += self.hidden_dim
                for i in range(1, self.num_layers):
                    self.conv.append(self.rgcn_func(self.hidden_dim, self.hidden_dim, self.num_relations).to(config.model_configuration['device']))
                    total_dim += self.hidden_dim
            else:
                self.fc0_5 = Linear(self.num_features, self.hidden_dim)
        else:
            if self.num_layers > 0:
                print("using layer specification and ignoring hidden_dim parameter.")
                print("layer_spec: " + str(self.layer_spec))
                self.conv.append(self.rgcn_func(self.num_features, self.layer_spec[0], self.num_relations).to(config.model_configuration['device']))
                total_dim += self.layer_spec[0]
                for i in range(1, self.num_layers):
                    self.conv.append(self.rgcn_func(self.layer_spec[i-1], self.layer_spec[i], self.num_relations).to(config.model_configuration['device']))
                    total_dim += self.layer_spec[i]

            else:
                self.fc0_5 = Linear(self.num_features, self.hidden_dim)
                total_dim += self.hidden_dim

        if self.pooling_type == "sagpool":
            self.pool1 = RGCNSAGPooling(total_dim, self.num_relations, ratio=config.model_configuration['pooling_ratio'], rgcn_func=config.model_configuration['conv_type'])
        elif self.pooling_type == "topk":
            self.pool1 = TopKPooling(total_dim, ratio=config.model_configuration['pooling_ratio'])

        self.fc1 = Linear(total_dim, self.lstm_dim1)
        
        if "lstm" in self.temporal_type:
            self.lstm = LSTM(self.lstm_dim1, self.lstm_dim2, batch_first=True)
            self.attn = Attention(self.lstm_dim2)
            self.lstm_decoder = LSTM(self.lstm_dim2, self.lstm_dim2, batch_first=True)
        else:
            self.fc1_5 = Linear(self.lstm_dim1, self.lstm_dim2)

        self.fc2 = Linear(self.lstm_dim2, self.num_classes)
        self.conv = nn.ModuleList(self.conv)


    def forward(self, x, edge_index, edge_attr, batch=None):
        attn_weights = dict()
        outputs = []
        if self.num_layers > 0:
            for i in range(self.num_layers):
                x = self.activation(self.conv[i](x, edge_index, edge_attr))
                x = F.dropout(x, self.dropout, training=self.training)
                outputs.append(x)
            x = torch.cat(outputs, dim=-1)
        else:
            x = self.activation(self.fc0_5(x))

        if self.pooling_type == "sagpool":
            x, edge_index, _, attn_weights['batch'], attn_weights['pool_perm'], attn_weights['pool_score'] = self.pool1(x, edge_index, edge_attr=edge_attr, batch=batch)
        elif self.pooling_type == "topk":
            x, edge_index, _, attn_weights['batch'], attn_weights['pool_perm'], attn_weights['pool_score'] = self.pool1(x, edge_index, edge_attr=edge_attr, batch=batch)
        else: 
            attn_weights['batch'] = batch

        if self.readout_type == "add":
            x = global_add_pool(x, attn_weights['batch'])
        elif self.readout_type == "mean":
            x = global_mean_pool(x, attn_weights['batch'])
        elif self.readout_type == "max":
            x = global_max_pool(x, attn_weights['batch'])
        else:
            pass

        x = self.activation(self.fc1(x))
    
        if self.temporal_type == "mean":
            x = self.activation(self.fc1_5(x.mean(axis=0)))
        elif self.temporal_type == "lstm_last":
            x_predicted, (h, c) = self.lstm(x.unsqueeze(0))
            x = h.flatten()
        elif self.temporal_type == "lstm_sum":
            x_predicted, (h, c) = self.lstm(x.unsqueeze(0))
            x = x_predicted.sum(dim=1).flatten()
        elif self.temporal_type == "lstm_attn":
            x_predicted, (h, c) = self.lstm(x.unsqueeze(0))
            x, attn_weights['lstm_attn_weights'] = self.attn(h.view(1,1,-1), x_predicted)
            x, (h_decoder, c_decoder) = self.lstm_decoder(x, (h, c))
            x = x.flatten()
        elif self.temporal_type == "lstm_seq": #used for step-by-step sequence prediction. 
            x_predicted, (h, c) = self.lstm(x.unsqueeze(0)) #x_predicted is sequence of predictions for each frame, h is hidden state of last item, c is last cell state
            x = x_predicted.squeeze(0) #we return x_predicted as we want to know the output of the LSTM for each value in the sequence
        else:
            pass
                
        return F.log_softmax(self.fc2(x), dim=-1), attn_weights