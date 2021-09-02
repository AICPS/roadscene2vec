import os
import sys

sys.path.append(os.path.dirname(sys.path[0]))
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, FastRGCNConv

'''Model skeleton for creating custom scene graph learning models'''
class CustomGraphModel(nn.Module):
    def __init__(self, config):
        super(CustomGraphModel, self).__init__()
        #problem specification settings
        self.num_features = config.model_configuration['num_of_classes'] #number of node features #TODO: change num_of_classes to num_features or node_features
        self.num_relations = config.model_configuration['num_relations'] #number of relation types.
        self.num_classes  = config.model_configuration['nclass'] #number of possible output classes.
        

        #graph learning model definition'
        self.num_layers = config.model_configuration['num_layers'] #defines number of graph learning layers.
        self.hidden_dim = config.model_configuration['hidden_dim'] #hidden size of each graph learning layer.
        self.layer_spec = None if config.model_configuration['layer_spec'] == None else list(map(int, config.model_configuration['layer_spec'].split(','))) #allows manual specification of each layer's size
        self.rgcn_func = FastRGCNConv if config.model_configuration['conv_type'] == "FastRGCNConv" else RGCNConv #define graph convolution operation performed
        self.activation = F.relu if config.model_configuration['activation'] == 'relu' else F.leaky_relu #define activation function used
        self.pooling_type = config.model_configuration['pooling_type'] #define type of graph pooling layer
        self.readout_type = config.model_configuration['readout_type'] #define type of graph readout operation

        #temporal modeling settings
        self.temporal_type = config.model_configuration['temporal_type'] #define type of temporal modeling used
        self.lstm_dim1 = config.model_configuration['lstm_input_dim']
        self.lstm_dim2 = config.model_configuration['lstm_output_dim']

        #regularizers
        self.dropout = config.model_configuration['dropout']

        #TODO: implement model construction

    def forward(self, edge_index, edge_attr, batch=None):
        pass #TODO: implement