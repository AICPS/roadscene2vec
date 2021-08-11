import sys, os
#import check_gpu as cg
#os.environ['CUDA_VISIBLE_DEVICES'] = cg.get_free_gpu()
sys.path.append(os.path.dirname(sys.path[0]))
from learning.util.image_trainer import Image_Trainer
from learning.util.scenegraph_trainer import Scenegraph_Trainer
import pandas as pd
from util.config_parser import configuration
import wandb

#python 3_train_model.py --yaml_path C:\users\harsi\research\sg2vec\sg2vec\config\learning_config.yaml  

def train_Trainer(learning_config, iterations=1):
    ''' Training the dynamic kg algorithm with different attention layer choice.'''

    #wandb setup 
    wandb_arg= wandb.init(project=learning_config.wandb_configuration['project'], entity=learning_config.wandb_configuration['entity'])
    ######
    outputs = []
    labels = []
    metrics = []
    
    categories_train_list = []
    categories_test_list = []

    for i in range(iterations):
        if learning_config.training_configuration["task_type"] == 'cnn_image_classification':
            trainer = Image_Trainer(learning_config, wandb_arg)
            trainer.split_dataset()
            trainer.build_model()
            trainer.learn()
            categories_train, categories_test, metric = trainer.eval_model()
            categories_train_list.append(categories_train)
            categories_test_list.append(categories_test)
            metrics.append(metric)
            
        elif learning_config.training_configuration["task_type"] in ['sequence_classification','graph_classification','collision_prediction']:
            trainer = Scenegraph_Trainer(learning_config, wandb_arg)
            trainer.split_dataset()
            trainer.build_model()
            trainer.learn()
            outputs_train, labels_train, outputs_test, labels_test, metric = trainer.evaluate()
            outputs += outputs_test
            labels  += labels_test
            metrics.append(metric)
        else:
            raise ValueError("Task unrecognized")
    
    trainer.save_model()

if __name__ == "__main__":
    # the entry of dynkg pipeline training
    learning_config = configuration(sys.argv[1:])
    train_Trainer(learning_config)