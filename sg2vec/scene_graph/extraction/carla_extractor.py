import sys, os
from pathlib import Path
sys.path.append(str(Path("../../")))
from sg2vec.scene_graph.nodes import Node
from sg2vec.scene_graph.nodes import ObjectNode
from sg2vec.scene_graph.extraction.extractor import Extractor as ex
from sg2vec.scene_graph.scene_graph import SceneGraph
import pickle as pkl
import yaml
import data.dataset as ds
from tqdm import tqdm

"""CarlaExtractor initializes relational settings and creates a CarlaSceneGraphSequenceGenerator object to extract scene graphs using raw scene data."""
class CarlaExtractor(ex):
    def __init__(self,config):
        super(CarlaExtractor, self).__init__(config)
    
        self.data_set = ds.GroundTruthDataset(self.conf)
        self.data_set.dataset_save_path = self.conf.location_data["input_path"]
        self.data_set = self.data_set.load()

        
        
    '''Load scenegraphs and store scenegraphs in the form {sequence{frame{scenegraph}} using the raw data given in the form {sequence{frame{raw_data}}'''
    def load(self):
        raw_data = self.data_set.raw_scenes
        sg_extracted = {}
        for sequence in tqdm(raw_data):
                sg_extracted[sequence] = {}
                try:
                    framedict = raw_data[sequence]
                    image_frames = list(framedict.keys()) #this is the list of frame names
                    image_frames = sorted(image_frames)
                    #### filling the gap between lane change where some of ego node might miss the invading lane information. ####
                    start_frame_number = 0; end_frame_number = 0; invading_lane_idx = None
                    
                    for idx, frame_number in enumerate(image_frames):
                        if "invading_lane" in framedict[str(frame_number)]['ego']:
                            start_frame_number = idx
                            invading_lane_idx = framedict[str(frame_number)]['ego']['invading_lane']
                            break

                    for frame_number in image_frames[::-1]:
                        if "invading_lane" in framedict[str(frame_number)]['ego']:
                            end_frame_number = image_frames.index(frame_number)
                            break
                
                    for idx in range(start_frame_number, end_frame_number):
                        framedict[str(image_frames[idx])]['ego']['invading_lane'] = invading_lane_idx
                    
                    for frame, frame_dict in framedict.items():
                        if str(frame) in image_frames: 
                            scenegraph = SceneGraph(self.relation_extractor, framedict = frame_dict, framenum = frame, platform = "carla")
                            sg_extracted[sequence][int(frame)] = scenegraph
                        
                except Exception as e:
                    import traceback
                    print("We have problem creating the Carla scenegraphs")
                    print(e)
                    traceback.print_exc()


        self.scene_graphs = sg_extracted

     
    '''Returns SceneGraphDataset object containing scengraphs, labels, action types, and meta data'''
    def getDataSet(self):
        try:
            sg_ds = ds.SceneGraphDataset(self.conf,self.scene_graphs,self.data_set.action_types,self.data_set.labels,self.data_set.meta)
            sg_ds.folder_names = self.data_set.folder_names
            return sg_ds
        #can just pass in self.dataset.conf but opting not to do so for clarity
        except Exception as e:
            import traceback
            print("We have problem creating scenegraph dataset object from the extracted Carla scenegraphs")
            print(e)
            traceback.print_exc()
    
    
    #remove this if breaks functionality or not needed
    def subsample(self, scenegraphs): 
        '''
            This function will subsample the original scenegraph sequence dataset (self.scenegraphs_sequence). 
            Before running this function, it includes a variant length of graph sequences. 
            We expect the length of graph sequences will be homogenenous after running this function.

            The default value of number_of_frames will be 20; Could be a tunnable hyperparameters.
        '''
        number_of_frames=self.framenum
        
        sequence = []
        frame_numbers = []
        acc_number = 0
        modulo = int(len(scenegraphs) / number_of_frames)
        if modulo == 0:
            modulo = 1

        for idx, (timeframe, scenegraph) in enumerate(scenegraphs.items()):
            if idx % modulo == 0 and acc_number < number_of_frames:
                sequence.append(scenegraph)
                frame_numbers.append(timeframe)
                acc_number+=1
        return sequence, frame_numbers
