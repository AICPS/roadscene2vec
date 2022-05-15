import os
import sys

sys.path.append(os.path.dirname(sys.path[0]))
from roadscene2vec.util.config_parser import configuration
import roadscene2vec
from scenegraph_trainer import Scenegraph_Trainer

sys.modules['util'] = roadscene2vec.util
import torch.nn as nn
import wandb


def train():
    training_config = configuration(r"use_case_4_config.yaml",from_function = True) 
    if training_config.wandb_configuration['project'] != None and training_config.wandb_configuration['entity'] != None:
        wandb_arg= wandb.init(project=training_config.wandb_configuration['project'], entity=training_config.wandb_configuration['entity'])
    else:
        wandb_arg = None
    trainer = Scenegraph_Trainer(training_config, wandb_arg)
    trainer.load_model() #load the proper model using the trainer
    trainer.loss_func = nn.CrossEntropyLoss() #set loss function
    trainer.build_transfer_learning_dataset()
    outputs_test, labels_test, metrics = trainer.evaluate_transfer_learning()    
    print(metrics) 

if __name__ == "__main__":
    train() #Assess risk of transfer dataset
    