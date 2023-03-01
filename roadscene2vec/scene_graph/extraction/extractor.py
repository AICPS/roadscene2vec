from abc import ABC
import roadscene2vec.scene_graph.relation_extractor as r_e


'''
This class defines the abstract base class of scene-graph extractors. scene-graph extractors can extract data from many different formats to generate SceneGraphDatasets.
'''
class Extractor(ABC):
    def __init__(self, config):
        self.conf = config
        self.dataset_type = self.conf.dataset_type
        self.scene_graphs = {}
        self.relation_extractor = r_e.RelationExtractor(config)
        self.framenum = self.conf.relation_extraction_settings["frames_limit"]
        