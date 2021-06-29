import sys, os
sys.path.append(os.path.dirname(sys.path[0]))
import sg2vec
# from sg2vec.data import real_preprocessor as ip
from sg2vec.util.config_parser import configuration
from sg2vec.scene_graph.extraction import image_extractor as RealEx
from sg2vec.util.visualizer import *

def im2graph():
    real_scenegraph_extraction_config = configuration(r"C:\Users\harsi\research\sg2vec\sg2vec\config\scene_graph_config_real.yaml",from_function = True) #replace with path to sg2vec\config\data_config_real.yaml
                                                                                                                                            #config should contain path to pkl containing preprocessed RawImageDataset
    real_scenegraph_extraction_object = RealEx.RealExtractor(real_scenegraph_extraction_config) #creating Real Image Scenegraph Extractor using config
    real_scenegraph_extraction_object.load() #extract scenegraphs for each frame in each sequence using Real Image Scenegraph Extractor
    real_scenegraph_dataset = real_scenegraph_extraction_object.getDataSet() #returned Scenegraph Dataset Object containing the extracted Real Image Scenegraphs for each frame in each sequence 
    return real_scenegraph_dataset

def extract_seq():
    real_data_preprocessing_config = configuration(r"C:\Users\harsi\research\sg2vec\sg2vec\config\data_config_real.yaml",from_function = True) #replace with path to sg2vec\config\scene_graph_config_real.yaml
                                                                                                                                  #config should contain path to folder containing sequences to evaluate
    real_preprocessor = ip.RealPreprocessor(real_data_preprocessing_config) #creating Real Image Preprocessor using config
    real_preprocessor.load() #preprocesses sequences by extracting frame data for each sequence
    real_dataset = real_preprocessor.getDataSet() #get RawImageDataset using Real Image Preprocessor
    real_dataset.save() #save RawImageDataset
    real_image_scenegraph_dataset = im2graph() #Extract scenegraphs for each frame for the each sequence 
    real_image_scenegraph_dataset.save() #save SceneGraphDataset

if __name__ == "__main__":
    extract_seq() 
    visualize(r"C:\Users\harsi\research\sg2vec\sg2vec\config\scene_graph_config_real.yaml") #visualize extracted scenegraphs using the stored location in the scene graph config