import sys, os, pdb
#import check_gpu as cg
#os.environ['CUDA_VISIBLE_DEVICES'] = cg.get_free_gpu()
sys.path.append(os.path.dirname(sys.path[0]))
from learning.util.image_trainer import Image_Trainer
from learning.util.scenegraph_trainer import Scenegraph_Trainer
import pandas as pd
from util.config_parser import configuration
import wandb

#python 4_test_model.py --yaml_path C:\users\harsi\research\roadscene2vec\roadscene2vec\config\learning_config.yaml  

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
        trainer.split_dataset() #in learning config, split_ratio should be 1 so that all data is put into test_data
        trainer.load_model()
        categories_train, categories_test, metric = trainer.eval_model()
        categories_train_list.append(categories_train)
        categories_test_list.append(categories_test)
        metrics.append(metric)
        
    elif learning_config.training_configuration["dataset_type"] == "scenegraph":
        trainer = Scenegraph_Trainer(learning_config, wandb_arg)
        trainer.split_dataset() #in learning config, split_ratio should be 1 so that all data is put into test_data
        trainer.load_model()
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
    test_Trainer(learning_config)