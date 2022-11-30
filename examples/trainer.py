import os
import sys
from pathlib import Path

sys.path.append(str(Path("../../")))
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random

from roadscene2vec.learning.model.cnn_lstm import CNN_LSTM_Classifier
from roadscene2vec.learning.model.lstm import LSTM_Classifier
from roadscene2vec.learning.model.mrgcn import MRGCN
from roadscene2vec.learning.model.mrgin import MRGIN
from roadscene2vec.learning.model.cnn import CNN_Classifier
from roadscene2vec.learning.model.resnet50_lstm import ResNet50_LSTM_Classifier
from roadscene2vec.learning.model.resnet50 import ResNet50_Classifier


'''Class implementing basic trainer functionality such as model building, saving, and loading.'''
class Trainer:

    def __init__(self, config, wandb_a = None):
        self.config = config
        if wandb_a != None:
            self.log = True
            self.wandb = wandb_a
            self.wandb_config = wandb_a.config
            # load config into wandb
            for section in self.config.args:
                for config_arg in self.config.args[section]:
                    self.wandb_config[section+'.'+config_arg] = self.config.args[section][config_arg]
        else:
            self.log = False             
        if self.config.training_configuration["seed"] != None: 
            self.config.seed = self.config.training_configuration["seed"]
            np.random.seed(self.config.seed)
            torch.manual_seed(self.config.seed)
        else:
            self.config.seed = random.randint(0,2**32) #default value
            np.random.seed(self.config.seed)
            torch.manual_seed(self.config.seed)
            
        self.toGPU = lambda x, dtype: torch.as_tensor(x, dtype=dtype, device=self.config.model_configuration['device'])
        self.initialize_best_metrics()


    #defines initial values for "best" metrics so that they can be updated during training by the model.
    def initialize_best_metrics(self):
        self.best_val_loss = 99999
        self.best_epoch = 0
        self.best_val_acc = 0
        self.best_val_auc = 0
        self.best_val_confusion = []
        self.best_val_f1 = 0
        self.best_val_mcc = -1.0
        self.best_val_acc_balanced = 0
        self.unique_clips = {}

    #abstract function implemented by subclasses
    def split_dataset(self):
        raise NotImplementedError


    def build_model(self): #this involves changing mrcgn and mrgin files to be compatible with new config tht we pass in
        # BD mode
        #self.config.num_features = len(self.feature_list)
        #self.config.num_relations = max([r.value for r in Relations])+1
        if self.config.model_configuration["model"] == "mrgcn":
            self.model = MRGCN(self.config).to(self.config.model_configuration["device"])
        elif self.config.model_configuration["model"]  == "mrgin":
            self.model = MRGIN(self.config).to(self.config.model_configuration["device"])
        elif self.config.model_configuration["model"]  == "cnn":
            self.model = CNN_Classifier((self.config.training_configuration['batch_size'], self.frame_limit,self.color_channels, self.im_height, self.im_width), self.config).to(self.config.model_configuration["device"])
        elif self.config.model_configuration["model"]  == "cnn_lstm":
            self.model = CNN_LSTM_Classifier((self.config.training_configuration['batch_size'], self.frame_limit,self.color_channels, self.im_height, self.im_width), self.config).to(self.config.model_configuration["device"])
        elif self.config.model_configuration["model"]  == "lstm":
            self.model = LSTM_Classifier((self.config.training_configuration['batch_size'], self.frame_limit,self.color_channels, self.im_height, self.im_width),'lstm', self.config).to(self.config.model_configuration["device"])        
        elif self.config.model_configuration["model"]  == "gru":
            self.model = LSTM_Classifier((self.config.training_configuration['batch_size'], self.frame_limit,self.color_channels, self.im_height, self.im_width), 'gru', self.config).to(self.config.model_configuration["device"]) 
        elif self.config.model_configuration["model"] == "resnet50_lstm":
            self.model = ResNet50_LSTM_Classifier((self.config.training_configuration['batch_size'], self.frame_limit,self.color_channels, self.im_height, self.im_width), self.config).to(self.config.model_configuration["device"]) 
        elif self.config.model_configuration["model"] == "resnet50":
            self.model = ResNet50_Classifier((self.config.training_configuration['batch_size'], self.frame_limit,self.color_channels, self.im_height, self.im_width), self.config).to(self.config.model_configuration["device"]) 
        else:
            raise Exception("model selection is invalid: " + self.config.model_configuration["model"])
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.training_configuration["learning_rate"], weight_decay=self.config.training_configuration["weight_decay"])
        
        if self.config.model_configuration["load_model"]  == False:
            if self.class_weights.shape[0] < 2:
                self.loss_func = nn.CrossEntropyLoss()
            else:
                self.loss_func = nn.CrossEntropyLoss(weight=self.class_weights.float().to(self.config.model_configuration["device"]))
     
            #wandb.watch(self.model, log="all")
            if self.log:
                self.wandb.watch(self.model, log="all")
        else:
            pass 

    # Pick between Standard Training and KFold Cross Validation Training
    def learn(self):  
        if self.config.training_configuration["n_fold"] <= 1 or self.config.location_data["transfer_path"] != None:
            print('\nRunning Standard Training Loop\n')
            self.train()
        else:
            print('\nRunning {}-Fold Cross Validation Training Loop\n'.format(self.config.training_configuration["n_fold"]))
            self.cross_valid()

    #abstract method implemented by subclasses
    def train(self):
        raise NotImplementedError

    def inference(self):
        raise NotImplementedError
    
    def cross_valid(self):
        raise NotImplementedError

    
    def save_model(self):
        """Function to save the model."""
        saved_path = Path(self.config.model_configuration["model_save_path"]).resolve()
        if os.path.dirname(saved_path) != '':
            os.makedirs(os.path.dirname(saved_path), exist_ok=True)
        torch.save(self.model.state_dict(), str(saved_path))
        with open(Path.joinpath(saved_path.parent, 'model_parameters.txt'), "w+") as f:
            f.write(str(self.config))
            f.write('\n')
            f.write(str(' '.join(sys.argv)))

    def load_model(self):
        """Function to load the model."""
        saved_path = Path(self.config.model_configuration["model_load_path"]).resolve()
        if saved_path.exists():
            self.build_model()
            self.model.load_state_dict(torch.load(str(saved_path)))
            self.model.eval()
        else:
            raise FileNotFoundError("Failed to load model. Model load path does not exist: " + str(saved_path))