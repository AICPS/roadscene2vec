# -*- coding: utf-8 -*-
import pdb
import os
import detectron2.utils.visualizer
import sys
from scene_graph.relation_extractor import ActorType, Relations, RELATION_COLORS
from scene_graph.scene_graph import SceneGraph
from scene_graph.image_scenegraph import RealSceneGraph
from detectron2.data import MetadataCatalog
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from glob import glob
import pandas as pd
import torch
import json
import pickle as pkl
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import math
import itertools
import networkx as nx
import random
import cv2
import numpy as np
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
# import some common libraries
matplotlib.use("Agg")
# import some common detectron2 utilities
sys.path.append(os.path.dirname(sys.path[0]))


def create_text_labels_with_idx(classes, scores, class_names):
    """
    Args:
        classes (list[int] or None):
        scores (list[float] or None):
        class_names (list[str] or None):

    Returns:
        list[str] or None
    """
    labels = None
    if classes is not None and class_names is not None and len(class_names) > 1:
        labels = [class_names[i] for i in classes]
    if scores is not None:
        if labels is None:
            labels = ["{:.0f}%".format(s * 100) for s in scores]
        else:
            labels = ["{}_{} {:.0f}%".format(
                l, idx, s * 100) for idx, (l, s) in enumerate(zip(labels, scores))]
    return labels


detectron2.utils.visualizer._create_text_labels = create_text_labels_with_idx


'''Generate scenegraphs for a given set of raw image sequences'''
class ImageSceneGraphSequenceGenerator:
    def __init__(self, framenum, cache_fname='real_dyngraph_embeddings.pkl', platform='image'):
        # [
        #   {'node_embeddings':..., 'edge_indexes':..., 'edge_attrs':..., 'label':...}
        # ]
        self.scenegraphs_sequence = {} #changed from [] to {}

        # cache_filename determine the name of caching file name storing self.scenegraphs_sequence and
        self.cache_filename = cache_fname

        # specify which type of data to load into model (options: image or honda)
        self.platfrom = platform

        # flag for turning on visualization
        self.visualize = False

        # config used for parsing CARLA:
        # this is the number of global classes defined in CARLA.
        self.num_classes = 8

        # gets a list of all feature labels (which will be used) for all scenegraphs
        # self.feature_list = {"rel_location_x",
        #                      "rel_location_y", #add 3 columns for relative vector values
        #                      "distance_abs", # adding absolute distance to ego
        #                     }
        self.feature_list = set()
        self.framenum = framenum
        # create 1hot class labels columns.
        for i in range(self.num_classes):
            self.feature_list.add("type_"+str(i))

        # detectron setup.
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        self.coco_class_names = MetadataCatalog.get(
            self.cfg.DATASETS.TRAIN[0]).get("thing_classes")
        self.predictor = DefaultPredictor(self.cfg)

    def cache_exists(self):
        return Path(self.cache_filename).exists()

    def load_from_cache(self):
        with open(self.cache_filename, 'rb') as f:
            self.scenegraphs_sequence, self.feature_list = pkl.load(f)

    '''Load scenegraphs using raw image frame tensors'''
    def load(self, seq_tensors): #seq_tensors[seq][frame/jpgname] = frame tensor
#         count = 0
        for sequence in seq_tensors:
#             if count == 3:
#                 break
#             count += 1
            self.scenegraphs_sequence[sequence] = {}

            for frame in seq_tensors[sequence]:

                out_img_path = None
                bounding_boxes = self.get_bounding_boxes(
                        seq_tensors[sequence][frame], out_img_path=out_img_path)
                
                scenegraph = RealSceneGraph(seq_tensors[sequence][frame], 
                                            bounding_boxes, 
                                            coco_class_names=self.coco_class_names, 
                                            platform=self.platfrom)
                                            
                self.scenegraphs_sequence[sequence][frame] = scenegraph

        return self.scenegraphs_sequence


    def cache_dataset(self, filename):
        with open(str(filename), 'wb') as f:
            pkl.dump((self.scenegraphs_sequence, self.feature_list), f)
            
            
    def get_bounding_boxes(self, img_tensor, out_img_path=None):
        im = img_tensor
        outputs = self.predictor(im)
        if out_img_path:
            # We can use `Visualizer` to draw the predictions on the image.
            v = detectron2.utils.visualizer.Visualizer(
                im[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            cv2.imwrite(out_img_path, out.get_image()[:, :, ::-1])

        # todo: after done scp to server
        # crop im to remove ego car's hood
        # find threshold then remove from pred_boxes, pred_classes, check image_size
        bounding_boxes = outputs["instances"].pred_boxes, outputs["instances"].pred_classes, outputs["instances"].image_size
        return bounding_boxes



    def process_graph_sequences(self, scenegraphs, frame_numbers, folder_name=None):
        '''
            The self.scenegraphs_sequence should be having same length after the subsampling. 
            This function will get the graph-related features (node embeddings, edge types, adjacency matrix) from scenegraphs.
            in tensor formats.
        '''
        sequence = []

        for idx, (scenegraph, frame_number) in enumerate(zip(scenegraphs, frame_numbers)):
            sg_dict = {}

            node_name2idx = {node: idx for idx,
                             node in enumerate(scenegraph.g.nodes)}

            sg_dict['node_features'] = self.get_node_embeddings(scenegraph)
            sg_dict['edge_index'], sg_dict['edge_attr'] = self.get_edge_embeddings(
                scenegraph, node_name2idx)
            sg_dict['folder_name'] = folder_name
            sg_dict['frame_number'] = frame_number
            sg_dict['node_order'] = node_name2idx
            sequence.append(sg_dict)

        # import pdb; pdb.set_trace()
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
                acc_number += 1

        return sequence, frame_numbers

    def get_node_embeddings(self, scenegraph):
        rows = []
        labels = []
        ego_attrs = None

        # extract ego attrs for creating relative features
        for node, data in scenegraph.g.nodes.items():
            if "ego" in str(node).lower():
                ego_attrs = data['attr']

        if ego_attrs == None:
            raise NameError("Ego not found in scenegraph")

        def get_embedding(node, row):
            # for key in self.feature_list:
            #     if key in node.attr:
            #         row[key] = node.attr[key]
            row['type_'+str(node.label.value)] = 1  # assign 1hot class label
            return row

        for idx, node in enumerate(scenegraph.g.nodes):
            d = defaultdict()
            row = get_embedding(node, d)
            labels.append(node.label.value)
            rows.append(row)

        embedding = pd.DataFrame(data=rows, columns=self.feature_list)
        embedding = embedding.fillna(value=0)  # fill in NaN with zeros
        embedding = torch.FloatTensor(embedding.values)

        return embedding

    def get_edge_embeddings(self, scenegraph, node_name2idx):
        edge_index = []
        edge_attr = []
        for src, dst, edge in scenegraph.g.edges(data=True):
            edge_index.append((node_name2idx[src], node_name2idx[dst]))
            edge_attr.append(edge['object'].value)

        edge_index = torch.transpose(torch.LongTensor(edge_index), 0, 1)
        edge_attr = torch.LongTensor(edge_attr)

        return edge_index, edge_attr

    def format_folders(self, all_video_clip_dirs):
        print('Begin formatting folders for Honda Dataset')
        for path in tqdm(all_video_clip_dirs):
            raw_path = path / "raw_images"
            raw_path.mkdir(exist_ok=True)
            honda_clips = [x for x in path.iterdir() if x.is_file()]
            for clip in honda_clips:
                if (clip.name.endswith(".jpg") or clip.name.endswith(".png")):
                    new_path = raw_path / clip.name
                    clip.replace(new_path)
        print('Finish formatting folders for Honda Dataset')
