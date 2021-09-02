import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

'''ResNet-50-based CNN classifier. Uses pre-trained ResNet-50 model weights to initialize the model.'''
class ResNet50_Classifier(nn.Module):

    def __init__(self, input_shape, cfg):
        super(ResNet50_Classifier, self).__init__()
        self.cfg = cfg
        self.batch_size, self.frames, self.channels, self.height, self.width = input_shape
        self.resnet = nn.Sequential(*list(models.resnet50(pretrained=True, progress=True).children())[:-1])
        self.dropout = self.cfg.model_configuration['dropout']
        if self.cfg.training_configuration['task_type'] == 'collision_prediction':
            self.l1 = nn.Linear(in_features=2048, out_features=512)
            self.l2 = nn.Linear(in_features=512, out_features=2)
        elif self.cfg.training_configuration['task_type'] == 'sequence_classification':
            self.l1 = nn.Linear(in_features=2048*self.frames, out_features=512)
            self.l2 = nn.Linear(in_features=512, out_features=2)
            self.TimeDistributed = lambda curr_layer, prev_layer : torch.stack([curr_layer(prev_layer[:,i]) for i in range(self.frames)], dim=1)

    def forward(self, x):
        if self.cfg.training_configuration['task_type'] == 'sequence_classification':
          x = self.TimeDistributed(self.resnet, x)
          x = torch.flatten(x, start_dim=1) #flatten resnet outputs for a whole sequence into a single vector.
        if self.cfg.training_configuration['task_type'] == 'collision_prediction':
          x = torch.cat([i for i in x])     #process each input image separately with resnet. 
          x = torch.squeeze(self.resnet(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.l1(x)
        x = F.dropout(F.relu(x), p=self.dropout, training=self.training)
        x = self.l2(x)
        return F.log_softmax(x, dim=-1)