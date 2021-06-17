# -*- coding: utf-8 -*-
# import some common libraries
import sys, os
import math
import matplotlib
from imageio.plugins._tifffile import sequence
matplotlib.use("Agg")
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import pickle as pkl
import json
import torch
import pandas as pd
from glob import glob
sys.path.append(os.path.dirname(sys.path[0]))
from scene_graph.scene_graph import SceneGraph
import pdb

'''Generate scenegraphs for a given set of Carla sequences'''
class CarlaSceneGraphSequenceGenerator:
    def __init__(self, framenum, cache_fname='dyngraph_embeddings.pkl'):
        # [ 
        #   {'node_embeddings':..., 'edge_indexes':..., 'edge_attrs':..., 'label':...}  
        # ]
        self.scenegraphs_sequence = []

        # cache_filename determine the name of caching file name storing self.scenegraphs_sequence and 
        self.cache_filename = cache_fname

        # flag for turning on visualization
        self.visualize = False
        
        # config used for parsing CARLA:
        # this is the number of global classes defined in CARLA.
        self.num_classes = 8
        
        # gets a list of all feature labels (which will be used) for all scenegraphs
        # self.feature_list = {"rel_location_x", 
        #                      "rel_location_y", 
        #                      "rel_location_z", #add 3 columns for relative vector values
        #                      "distance_abs", # adding absolute distance to ego
        #                     }
        self.feature_list = set()
        self.framenum = framenum
        # create 1hot class labels columns.
        for i in range(self.num_classes):
            self.feature_list.add("type_"+str(i))

    def cache_exists(self):
        return Path(self.cache_filename).exists()

    def load_from_cache(self):
        with open(self.cache_filename,'rb') as f: 
            self.scenegraphs_sequence , self.feature_list = pkl.load(f)


    '''Load scenegraphs and store scenegraphs in the form {sequence{frame{scenegraph}} using the raw data given in the form {sequence{frame{raw_data}}'''
    def load(self, raw_data):
        sg_extracted = defaultdict()
        for sequence in raw_data:
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
                            scenegraph = SceneGraph(frame_dict, framenum=frame)
                            sg_extracted[sequence][int(frame)] = scenegraph
                        
                except Exception as e:
                    import traceback
                    print("We have problem creating the Carla scenegraphs")
                    print(e)
                    traceback.print_exc()


        return sg_extracted

        
        
    def cache_dataset(self, filename):
        with open(str(filename), 'wb') as f:
            pkl.dump((self.scenegraphs_sequence, self.feature_list), f)
            
    def process_graph_sequences(self, scenegraphs, frame_numbers, folder_name=None):
        '''
            The self.scenegraphs_sequence should be having same length after the subsampling. 
            This function will get the graph-related features (node embeddings, edge types, adjacency matrix) from scenegraphs.
            in tensor formats.
        '''
        sequence = []

        for idx, (scenegraph, frame_number) in enumerate(zip(scenegraphs, frame_numbers)):
            sg_dict = {}
            
            node_name2idx = {node:idx for idx, node in enumerate(scenegraph.g.nodes)}

            sg_dict['node_features']                    = self.get_node_embeddings(scenegraph)
            sg_dict['edge_index'], sg_dict['edge_attr'] = self.get_edge_embeddings(scenegraph, node_name2idx)
            sg_dict['folder_name'] = folder_name
            sg_dict['frame_number'] = frame_number
            sg_dict['node_order'] = node_name2idx
            # import pdb; pdb.set_trace()
            sequence.append(sg_dict)

        return sequence

    def visualize_scenegraphs(self, clip_ids):
        self.visualize = True
        self.clip_ids = clip_ids

    def subsample(self, scenegraphs, number_of_frames=20): 
        '''
            This function will subsample the original scenegraph sequence dataset (self.scenegraphs_sequence). 
            Before running this function, it includes a variant length of graph sequences. 
            We expect the length of graph sequences will be homogenenous after running this function.

            The default value of number_of_frames will be 20; Could be a tunnable hyperparameters.
        '''
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
        
    def get_node_embeddings(self, scenegraph):
        rows = []
        labels=[]
        ego_attrs = None
        
        #extract ego attrs for creating relative features
        for node, data in scenegraph.g.nodes.items():
            if "ego:" in str(node):
                ego_attrs = data['attr']
        if ego_attrs == None:
            raise NameError("Ego not found in scenegraph")

        #rotating axes to align with ego. yaw axis is the primary rotation axis in vehicles
        ego_yaw = math.radians(ego_attrs['rotation'][0])
        cos_term = math.cos(ego_yaw)
        sin_term = math.sin(ego_yaw)

        def rotate_coords(x, y): 
            new_x = (x*cos_term) + (y*sin_term)
            new_y = ((-x)*sin_term) + (y*cos_term)
            return new_x, new_y
            
        def get_embedding(node, row):
            #subtract each vector from corresponding vector of ego to find delta         
            # if "location" in node.attr:
            #     ego_x, ego_y = rotate_coords(ego_attrs["location"][0], ego_attrs["location"][1])
            #     node_x, node_y = rotate_coords(node.attr["location"][0], node.attr["location"][1])
            #     row["rel_location_x"] = node_x - ego_x
            #     row["rel_location_y"] = node_y - ego_y
            #     row["rel_location_z"] = node.attr["location"][2] - ego_attrs["location"][2] #no axis rotation needed for Z
            #     row["distance_abs"] = math.sqrt(row["rel_location_x"]**2 + row["rel_location_y"]**2 + row["rel_location_z"]**2)

            row['type_'+str(node.type)] = 1 #assign 1hot class label
            return row
        
        for idx, node in enumerate(scenegraph.g.nodes):
            d = defaultdict()
            row = get_embedding(node, d)
            labels.append(node.type)
            rows.append(row)
            
        embedding = pd.DataFrame(data=rows, columns=self.feature_list)
        embedding = embedding.fillna(value=0) #fill in NaN with zeros
        embedding = torch.FloatTensor(embedding.values)
        
        return embedding

    def get_edge_embeddings(self, scenegraph, node_name2idx):
        edge_index = []
        edge_attr = []
        for src, dst, edge in scenegraph.g.edges(data=True):
            edge_index.append((node_name2idx[src], node_name2idx[dst]))
            edge_attr.append(edge['object'].value)

        edge_index = torch.transpose(torch.LongTensor(edge_index), 0, 1)
        edge_attr  = torch.LongTensor(edge_attr)
        
        return edge_index, edge_attr