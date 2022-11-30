import os
import sys

sys.path.append(os.path.dirname(sys.path[0]))
from learning.util.image_trainer import Image_Trainer
from learning.util.scenegraph_trainer import Scenegraph_Trainer
from util.config_parser import configuration
import wandb
import torch.nn as nn

#python 4_test_model.py --yaml_path ../config/graph_learning_config.yaml 

def test_Trainer(learning_config):
    ''' Training the dynamic kg algorithm with different attention layer choice.'''

    #wandb setup 
    wandb_arg= wandb.init(project=learning_config.wandb_configuration['project'], entity=learning_config.wandb_configuration['entity'])
    outputs = []
    labels = []
    metrics = []
    
    categories_train_list = []
    categories_test_list = []

    if learning_config.training_configuration["dataset_type"] == "real":
        trainer = Image_Trainer(learning_config, wandb_arg)
        trainer.split_dataset() 
        trainer.load_model() #set load model to true in config, and specify load path
        trainer.loss_func = nn.CrossEntropyLoss() #set loss function
        categories_train, categories_test, metric = trainer.eval_model()
        categories_train_list.append(categories_train)
        categories_test_list.append(categories_test)
        metrics.append(metric)
        
    elif learning_config.training_configuration["dataset_type"] == "scenegraph":
        trainer = Scenegraph_Trainer(learning_config, wandb_arg)
        trainer.split_dataset() 
        trainer.load_model() #set load model to true in config, and specify load path
        trainer.loss_func = nn.CrossEntropyLoss() #set loss function
        outputs_train, labels_train, outputs_test, labels_test, metric = trainer.evaluate()
        outputs += outputs_test
        labels  += labels_test
        metrics.append(metric)
    else:
        raise ValueError("Type unrecognized")


if __name__ == "__main__":
    # the entry of dynkg pipeline training
    learning_config = configuration(sys.argv[1:])
    test_Trainer(learning_config)