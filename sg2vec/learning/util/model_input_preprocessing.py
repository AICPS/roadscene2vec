#this is for cnn input using real images
import numpy as np
import pandas as pd
import torch
import math
from collections import defaultdict

def process_cnn_image_data(sequence, color_channels, im_height, im_width):
    seq_data = []
    # seq is seq num
    for frame_data in sequence.values(): # yields 
        #frame_data is dictionary array(h,w,rgb)
        flip_frame_data = np.zeros((color_channels, im_height, im_width))
        for h in range(frame_data.shape[0]):
            for w in range(frame_data.shape[1]):
                for rgb in range(frame_data.shape[2]):
                    flip_frame_data[rgb][h][w] = frame_data[h][w][rgb]
        seq_data.append(flip_frame_data)
    return seq_data



# this is for creation of trainer input using carla data 
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

        sg_dict['node_features']                    = get_carla_node_embeddings(scenegraph, feature_list)
        sg_dict['edge_index'], sg_dict['edge_attr'] = get_carla_edge_embeddings(scenegraph, node_name2idx)
        sg_dict['folder_name'] = folder_name
        sg_dict['frame_number'] = frame_number
        sg_dict['node_order'] = node_name2idx
        sequence.append(sg_dict)

    return sequence

def get_carla_node_embeddings(scenegraph, feature_list):
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
        
    def get_carla_embedding(node, row):
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
        row = get_carla_embedding(node, d)
        labels.append(node.type)
        rows.append(row)
        
    embedding = pd.DataFrame(data=rows, columns=feature_list)
    embedding = embedding.fillna(value=0) #fill in NaN with zeros
    embedding = torch.FloatTensor(embedding.values)
    
    return embedding


def get_carla_edge_embeddings(scenegraph, node_name2idx):
    edge_index = []
    edge_attr = []
    for src, dst, edge in scenegraph.g.edges(data=True):
        edge_index.append((node_name2idx[src], node_name2idx[dst]))
        edge_attr.append(edge['object'].value)

    edge_index = torch.transpose(torch.LongTensor(edge_index), 0, 1)
    edge_attr  = torch.LongTensor(edge_attr)
    
    return edge_index, edge_attr

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

        sg_dict['node_features'] = get_real_image_node_embeddings(scenegraph, feature_list)
        sg_dict['edge_index'], sg_dict['edge_attr'] = get_real_image_edge_embeddings(
            scenegraph, node_name2idx)
        sg_dict['folder_name'] = folder_name
        sg_dict['frame_number'] = frame_number
        sg_dict['node_order'] = node_name2idx
        sequence.append(sg_dict)

    return sequence

def get_real_image_node_embeddings(scenegraph, feature_list):
        rows = []
        labels = []
        ego_attrs = None

        # extract ego attrs for creating relative features
        for node, data in scenegraph.g.nodes.items():
            if "ego" in str(node).lower():
                ego_attrs = data['attr']

        if ego_attrs == None:
            raise NameError("Ego not found in scenegraph")

        def get_real_embedding(node, row):
            # for key in self.feature_list:
            #     if key in node.attr:
            #         row[key] = node.attr[key]
            row['type_'+str(node.label.value)] = 1  # assign 1hot class label
            return row

        for idx, node in enumerate(scenegraph.g.nodes):
            d = defaultdict()
            row = get_real_embedding(node, d)
            labels.append(node.label.value)
            rows.append(row)

        embedding = pd.DataFrame(data=rows, columns=feature_list)
        embedding = embedding.fillna(value=0)  # fill in NaN with zeros
        embedding = torch.FloatTensor(embedding.values)

        return embedding

def get_real_image_edge_embeddings(scenegraph, node_name2idx):
    edge_index = []
    edge_attr = []
    for src, dst, edge in scenegraph.g.edges(data=True):
        edge_index.append((node_name2idx[src], node_name2idx[dst]))
        edge_attr.append(edge['object'].value)

    edge_index = torch.transpose(torch.LongTensor(edge_index), 0, 1)
    edge_attr = torch.LongTensor(edge_attr)

    return edge_index, edge_attr




#==================================================================
