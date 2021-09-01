import sys, os
sys.path.append(os.path.dirname(sys.path[0]))
# import roadscene2vec.data.real_preprocessor as ip
from roadscene2vec.util.config_parser import configuration
# import roadscene2vec.scene_graph.extraction.image_extractor as RealEx
import roadscene2vec
from roadscene2vec.learning.util.trainer import Scenegraph_Trainer
from roadscene2vec.data.dataset import SceneGraphDataset
sys.modules['util'] = roadscene2vec.util
import wandb



def train():
    training_config = configuration(r"use_case_4_config.yaml",from_function = True) #replace with path to roadscene2vec\config\learning_config.yaml
    wandb_arg= wandb.init(project=training_config.wandb_configuration['project'], entity=training_config.wandb_configuration['entity'])
    trainer = Scenegraph_Trainer(training_config, wandb_arg)
    trainer.split_dataset()
    trainer.build_model()
    trainer.learn()
    trainer.build_transfer_learning_dataset()
    outputs_test, labels_test, metrics = trainer.evaluate_transfer_learning()    
    print(metrics) #prints dict format is that fine?

if __name__ == "__main__":
    train() #Assess risk of...
    
    #path should be to a folder containing one sequence is that okay? 
    #change comments in risk collison_pred
    