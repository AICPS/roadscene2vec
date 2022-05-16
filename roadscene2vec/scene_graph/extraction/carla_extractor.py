import sys
from pathlib import Path
sys.path.append(str(Path("../../")))
#from roadscene2vec.scene_graph.nodes import ObjectNode
from roadscene2vec.scene_graph.extraction.extractor import Extractor as ex
from roadscene2vec.scene_graph.scene_graph import SceneGraph
import roadscene2vec.data.dataset as ds
from tqdm import tqdm
import ast
from glob import glob
import json

"""CarlaExtractor initializes relational settings and creates a CarlaSceneGraphSequenceGenerator object to extract scene graphs using raw scene data."""
class CarlaExtractor(ex):
    def __init__(self,config):
        super(CarlaExtractor, self).__init__(config)
        
        self.input_path = self.conf.location_data['input_path']
        self.dataset = ds.SceneGraphDataset(self.conf)

        
        
    '''Load scenegraphs and store scenegraphs in the form {sequence{frame{scenegraph}} using the raw data given in the form {sequence{frame{raw_data}}'''
    def load(self):
        all_sequence_dirs = [x for x in Path(self.input_path).iterdir() if x.is_dir()]
        all_sequence_dirs = sorted(all_sequence_dirs, key=lambda x: int(x.stem.split('_')[0]))  
        self.dataset.folder_names = [path.stem for path in all_sequence_dirs]
        sg_extracted = {}
        for path in tqdm(all_sequence_dirs):
              seq = int(path.stem.split('_')[0])
              self.dataset.action_types[seq] = path.stem.split('_')[1]
              label_path = (path/"label.txt").resolve()
              metadata_path = (path/"metadata.txt").resolve()
  
              if label_path.exists():
                  with open(str(path/'label.txt'), 'r') as label_file:
                      lines = label_file.readlines()
                      l0 = 1.0 if float(lines[0].strip().split(",")[0]) >= 0 else 0.0 
                      self.dataset.labels[seq] = l0 
  
                
              if not metadata_path.exists():
                  raise FileNotFoundError((path/'metadata.txt').resolve())
              else:
                  with open(str(path/'metadata.txt'), 'r') as md_file:
                      md = md_file.read()
                      self.dataset.meta[seq] = ast.literal_eval(md)
                    
              txt_path = sorted(list(glob("%s/**/*.json" % str(path/"scene_raw"), recursive=True)))[0]
              with open(txt_path, 'r') as scene_dict_f:
                  try:
                      #self.dataset.raw_scenes[seq] = json.loads(scene_dict_f.read()) 
                      sg_extracted[seq] = {}
                      framedict = json.loads(scene_dict_f.read()) 
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
                              sg_extracted[seq][int(frame)] = scenegraph
                      if self.framenum != None:
                        sg_extracted[seq] = self.subsample(sg_extracted[seq])
                  except Exception as e:
                      import traceback
                      print("We have problem creating the Carla scenegraphs")
                      print(e)
                      traceback.print_exc()
                  
        self.dataset.scene_graphs = sg_extracted

     
    '''Returns SceneGraphDataset object containing scengraphs, labels, action types, and meta data'''
    def getDataSet(self):
        try:
            return self.dataset
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
        
        sequence = {}
        #frame_numbers = []
        acc_number = 0
        modulo = int(len(scenegraphs) / number_of_frames)
        if modulo == 0:
            modulo = 1

        for idx, (timeframe, scenegraph) in enumerate(scenegraphs.items()):
            if idx % modulo == 0 and acc_number < number_of_frames:
                sequence[timeframe] = scenegraph
                #sequence.append(scenegraph)
                #frame_numbers.append(timeframe)
                acc_number+=1
        return sequence
