import scene_graph.relation_extractor as r_e
import scene_graph.scene_graph as sg
import pickle as pkl

'''Base class used to create CarlaPreprocessor and RealPreprocessor'''
class Preprocessor():
    #abstract base class for preprocessors
    def __init__(self, config):
        self.conf = config
        self.dataset = None

