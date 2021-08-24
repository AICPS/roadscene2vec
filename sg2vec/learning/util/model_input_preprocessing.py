#this is for cnn input using real images
import numpy as np
import pandas as pd
import torch
import math
from collections import defaultdict


# this is for creation of trainer input using carla data #TODO move this to scenegraph dataset class?
#=======================================================
def process_carla_graph_sequences(scenegraphs, feature_list, frame_numbers = None, folder_name=None): #returns a dictionary containing sg metadata for each frame in a sequence
                                                        #default frame_numbers to len of sg dict that contains scenegraphs for each frame of the given sequence
    '''
        The self.scenegraphs_sequence should be having same length after the subsampling. 
        This function will get the graph-related features (node embeddings, edge types, adjacency matrix) from scenegraphs.
        in tensor formats.
    '''
    if frame_numbers == None:
        frame_numbers = sorted(list(scenegraphs.keys()))
    scenegraphs = [scenegraphs[frames] for frames in sorted(scenegraphs.keys())]
    sequence = []
    for idx, (scenegraph, frame_number) in enumerate(zip(scenegraphs, frame_numbers)):
        sg_dict = {}
        
        node_name2idx = {node:idx for idx, node in enumerate(scenegraph.g.nodes)}

        sg_dict['node_features']                    = scenegraph.get_carla_node_embeddings(feature_list)
        sg_dict['edge_index'], sg_dict['edge_attr'] = scenegraph.get_carla_edge_embeddings(node_name2idx)
        sg_dict['folder_name'] = folder_name
        sg_dict['frame_number'] = frame_number
        sg_dict['node_order'] = node_name2idx
        sequence.append(sg_dict)

    return sequence

#===================================================================

# this is for creation of trainer input using image data 
#===================================================================

def process_real_image_graph_sequences(scenegraphs, feature_list, frame_numbers=None, folder_name=None):
    '''
        The self.scenegraphs_sequence should be having same length after the subsampling. 
        This function will get the graph-related features (node embeddings, edge types, adjacency matrix) from scenegraphs.
        in tensor formats.
    '''
    if frame_numbers == None:
        frame_numbers = sorted(list(scenegraphs.keys()))
    scenegraphs = [scenegraphs[frames] for frames in sorted(scenegraphs.keys())]
    sequence = []

    for idx, (scenegraph, frame_number) in enumerate(zip(scenegraphs, frame_numbers)):
        sg_dict = {}

        node_name2idx = {node: idx for idx,
                         node in enumerate(scenegraph.g.nodes)}

        sg_dict['node_features'] = scenegraph.get_real_image_node_embeddings(feature_list)
        sg_dict['edge_index'], sg_dict['edge_attr'] = scenegraph.get_real_image_edge_embeddings(node_name2idx)
        sg_dict['folder_name'] = folder_name
        sg_dict['frame_number'] = frame_number
        sg_dict['node_order'] = node_name2idx
        sequence.append(sg_dict)

    return sequence



#==================================================================
