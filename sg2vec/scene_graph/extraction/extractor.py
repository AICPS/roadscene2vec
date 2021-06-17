from abc import ABC

import scene_graph.relation_extractor as r_e


'''
This class defines the abstract base class of scene-graph extractors. scene-graph extractors can extract data from many different formats to generate SceneGraphDatasets.
'''
import scene_graph.relation_extractor as r_e
import scene_graph.scene_graph as sg
import pickle as pkl

'''Base class used to create CarlaExtractor and RealExtractor'''
class Extractor(ABC):
    #abstract base class for preprocessors
    def __init__(self, config):
        self.conf = config
        self.dataset_type = self.conf.dataset_type
        self.scene_graphs = {}
        
        self.relation_extractor = r_e.RelationExtractor(config)

        self.framenum = self.conf.relation_extraction_settings["framenum"]
        