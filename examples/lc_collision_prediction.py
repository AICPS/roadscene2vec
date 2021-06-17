from sg2vec.data.dataset import *
from sg2vec.scene_graph import *
from sg2vec.learning import *

### keep for developement.
import sys, os
sys.path.append(os.path.dirname(sys.path[0]))


class Config:
    def __init__(self):
        pass




def lc_collision_pred(config):
    ''' main routine of the lc collision prediction pipeline '''
    
    raw_image_dataset  = extract_dataset(config) #extract dataset
    scene_graph_dataset= extract_sg_dataset(raw_image_dataset) # extract sg #combine sg and labels
    
    trainer = Trainer(scene_graph_dataset, config)
    
    ### conduct training here ###
    
    ### run the evaluation for testing set
    results = None
    
    visualize(config, results)
    store_results(config, results)

def extract_dataset(config):
    #any preprocessing steps?
    dataset = RawImageDataset(config)
    dataset.load()
    return dataset

def extract_sg_dataset(raw_image_dataset):
    # starting from RawImageDataset and finish with generating a SceneGraphDataset.
    pass

def visualize(config, results):
    pass

def store_results(config, results):
    pass


if __name__ == "__main__":
    config = None ##TODO: read the yaml file?
    lc_collision_pred(config)