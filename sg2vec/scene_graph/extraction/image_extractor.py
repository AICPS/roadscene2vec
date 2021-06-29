import os, sys, yaml
import pickle as pkl
import cv2
import detectron2
import sys, os
from pathlib import Path
sys.path.append(str(Path("../../")))
import sg2vec.data.dataset as ds
from sg2vec.scene_graph.extraction.extractor import Extractor as ex
from sg2vec.scene_graph.scene_graph import SceneGraph

from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.utils import visualizer 
from detectron2.config import get_cfg
from detectron2 import model_zoo
from sg2vec.scene_graph.extraction.bev import bev
from tqdm import tqdm

'''RealExtractor initializes relational settings and creates an ImageSceneGraphSequenceGenerator object to extract scene graphs using raw image data.'''
class RealExtractor(ex):
    #TODO: RealPreprocessor preprocessor 
    def __init__(self, config):
        super(RealExtractor, self).__init__(config) 
        
        self.data_set = ds.RawImageDataset()
        self.data_set.dataset_save_path = self.conf.location_data['input_path']
        self.data_set = self.data_set.load()
        
        # detectron setup
        model_path = 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file(model_path))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_path)
        self.coco_class_names = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).get('thing_classes')
        self.predictor = DefaultPredictor(self.cfg)

        # bev setup
        self.bev = bev.BEV(config.image_setttings['BEV_PATH'], mode='deploy')


    '''Load scenegraphs using raw image frame tensors'''
    def load(self): #seq_tensors[seq][frame/jpgname] = frame tensor
        try:
            seq_tensors = self.data_set.data
            scenegraphs_sequence = {}
            for sequence in tqdm(seq_tensors):
                scenegraphs_sequence[sequence] = {}
    
                for frame in seq_tensors[sequence]:
                    out_img_path = None
                    bounding_boxes = self.get_bounding_boxes(seq_tensors[sequence][frame], #seq_tensors[sequence][frame].numpy()
                                                             out_img_path=out_img_path)
                    
                    scenegraph = SceneGraph(self.relation_extractor,    
                                                bounding_boxes = bounding_boxes, 
                                                bev = self.bev,
                                                coco_class_names=self.coco_class_names, 
                                                platform=self.dataset_type)

                    scenegraphs_sequence[sequence][frame] = scenegraph
#                 break;
#                 import pdb; pdb.set_trace()
            self.scene_graphs = scenegraphs_sequence
        except Exception as e:
            import traceback
            print('We have problem creating the real image scenegraphs')
            print(e)
            traceback.print_exc()
        
    def get_bounding_box_annotated_image(self, im):
        v = visualizer.Visualizer(im[:, :, ::-1], 
            MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), 
            scale=1.2)
        out = v.draw_instance_predictions(self.predictor(im)['instances'].to('cpu'))
        return out.get_image()[:, :, ::-1]
            
    def get_bounding_boxes(self, img_tensor, out_img_path=None):
        im = img_tensor
        outputs = self.predictor(im)
        if out_img_path:
            # We can use `Visualizer` to draw the predictions on the image.
            out = self.get_bounding_box_annotated_image(im)
            cv2.imwrite(out_img_path, out)

        # todo: after done scp to server
        # crop im to remove ego car's hood
        # find threshold then remove from pred_boxes, pred_classes, check image_size
        bounding_boxes = outputs['instances'].pred_boxes, outputs['instances'].pred_classes, outputs['instances'].image_size
        return bounding_boxes

    
    '''Returns SceneGraphDataset object containing scengraphs, labels, and action types'''
    def getDataSet(self):
        try:
            sg_ds = ds.SceneGraphDataset(self.conf,self.scene_graphs,self.data_set.action_types,self.data_set.labels)
            sg_ds.folder_names = self.data_set.folder_names
            return sg_ds
        except Exception as e:
            import traceback
            print('We have problem creating scenegraph dataset object from the extracted real image scenegraphs')
            print(e)
            traceback.print_exc()
    
   
    
            
