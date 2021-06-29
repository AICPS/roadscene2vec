import sys, os
from pathlib import Path
sys.path.append(str(Path("../../")))
import sg2vec.scene_graph.relation_extractor as r_e
import sg2vec.scene_graph.scene_graph as sg
import pickle as pkl

'''Base class used to create CarlaPreprocessor and RealPreprocessor'''
class Preprocessor():
    #abstract base class for preprocessors
    def __init__(self, config):
        self.conf = config
        self.dataset = None

