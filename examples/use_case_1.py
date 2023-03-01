import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))
from roadscene2vec.util.config_parser import configuration
from roadscene2vec.scene_graph.extraction import image_extractor as RealEx
from roadscene2vec.scene_graph.extraction import carla_extractor as CarlaEx
from roadscene2vec.util.visualizer import visualize


def extract_seq(scenegraph_extraction_config):                                                                                          
  if scenegraph_extraction_config.dataset_type == "image":
    extract_img_seq(scenegraph_extraction_config)
  elif scenegraph_extraction_config.dataset_type == "carla":
    extract_carla_seq(scenegraph_extraction_config)
  else:
    raise ValueError("Extraction dataset type not recognized")
    
def extract_img_seq(scenegraph_extraction_config):                                                                                          
    sg_extraction_object = RealEx.RealExtractor(scenegraph_extraction_config) #creating Real Image Extractor using config
    sg_extraction_object.load() #preprocesses sequences by extracting frame data for each sequence
    scenegraph_dataset = sg_extraction_object.getDataSet() #returned scenegraphs from extraction
    scenegraph_dataset.save() #save SceneGraphDataset
    
def extract_carla_seq(scenegraph_extraction_config):                                                                                          
    sg_extraction_object = CarlaEx.CarlaExtractor(scenegraph_extraction_config) #creating Carla Extractor using config
    sg_extraction_object.load() #preprocesses sequences by extracting frame data for each sequence
    scenegraph_dataset = sg_extraction_object.getDataSet() #returned scenegraphs from extraction
    scenegraph_dataset.save() #save SceneGraphDataset

if __name__ == "__main__":
    scenegraph_extraction_config = configuration(r"use_case_1_scenegraph_extraction_config.yaml",from_function = True) #create scenegraph extraction config object
    extract_seq(scenegraph_extraction_config)  
    visualize(scenegraph_extraction_config) #visualize extracted scenegraphs
 
    