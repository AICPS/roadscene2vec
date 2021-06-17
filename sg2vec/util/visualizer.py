import os
import sys
import cv2
from PIL import Image
from io import BytesIO
from pprint import pprint
from networkx.drawing import nx_agraph, nx_pydot
sys.path.append(os.path.dirname(sys.path[0]))
from util import config_parser
from scene_graph.scene_graph import SceneGraph
# from scene_graph.extraction.carla_extractor import CarlaExtractor # going to ignore for now...
from scene_graph.extraction.image_extractor import RealExtractor

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

TEMP_PATH = '../config/scene_graph_config_real.yaml'

from timeit import default_timer as timer
def elapsed_time(func, *args, **kwargs):
  start = timer()
  output = func(*args, **kwargs)
  end = timer()
  print(f'{end - start} seconds elapsed.')
  return output

# Utilities
def get_parser(yml_path):
  return config_parser.configuration(yml_path)

def get_extractor(fname=TEMP_PATH):
  return RealExtractor(get_parser(fname))

def get_data(extractor):
  return extractor.data_set.data

def get_bev(extractor):
  return extractor.bev#.warpPerspective(frame)

def get_bbox(extractor, frame):
  return extractor.get_bounding_boxes(frame)

def get_scenegraph(extractor, bbox, bev):
  scenegraph = SceneGraph(extractor.relation_extractor,   
                          bounding_boxes=bbox, 
                          bev=bev,
                          coco_class_names=extractor.coco_class_names, 
                          platform=extractor.dataset_type)
  return scenegraph.g

def inspect_nodes(sg):
  for node in sg.nodes: print(node.name, end=' '); pprint(node.attr);

def inspect_relations(sg):
  for edge in sg.edges(data=True, keys=True): pprint(edge);

def yield_data(data):
  for sequence in data:
      for frame in data[sequence]:
        yield data[sequence][frame]

# Visualization
def cv2_color(frame):
  return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def draw_bbox(extractor, frame):
  return extractor.get_bounding_box_annotated_image(frame)

def draw_bev(bev, frame):
  img = bev.offset_image_height(frame)
  return bev.warpPerspective(img)
  
def draw_scenegraph_agraph(sg):
  # Not sure why this function cannot draw multi-edge connections to same node
  A = nx_agraph.to_agraph(sg) 
  A.layout('dot') 
  img = A.draw(format='png')
  return Image.open(BytesIO(img))

def draw_scenegraph_pydot(sg):
  A = nx_pydot.to_pydot(sg)
  img = A.create_png()
  return Image.open(BytesIO(img))

def draw(extractor, frame, bbox, bev, sg, save_path=None):
  
  img = frame
  plt.subplot(2, 3, 1)
  plt.imshow(cv2_color(img))
  plt.title("Raw Image")
  plt.axis('off')
  
  bbox_img = draw_bbox(extractor, frame)
  plt.subplot(2, 3, 2)
  plt.imshow(cv2_color(bbox_img))
  plt.title("Object Detection Image")
  plt.axis('off')
  
  bev_img = draw_bev(bev, frame)
  plt.subplot(2, 3, 3)
  plt.imshow(cv2_color(bev_img))
  plt.title("Bird's Eye Image")
  plt.axis('off')

  sg_img = draw_scenegraph_pydot(sg).convert('RGB')
  plt.subplot(2, 1, 2)
  plt.imshow(sg_img)
  plt.title("SceneGraph Image")
  plt.axis('off')

  # This call is slow!
  if save_path is not None: 
    plt.savefig(save_path, dpi=600)

  plt.show()
  
def visualize(sg_config_file):
    extractor = elapsed_time(get_extractor, sg_config_file)
    data = get_data(extractor)
    gen_data = yield_data(data)
    
    while True:
      try:
        frame = next(gen_data)
      except:
        print('- finished'); break;
      else:
        bbox = get_bbox(extractor, frame)
        bev = get_bev(extractor)
        sg = get_scenegraph(extractor, bbox, bev)
    
        draw(extractor, frame, bbox, bev, sg, save_path='output.png')
