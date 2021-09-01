import os
import sys
import cv2
from pathlib import Path
from PIL import Image
from io import BytesIO
from pprint import pprint
from networkx.drawing import nx_agraph, nx_pydot
sys.path.append(str(Path("../../")))
from roadscene2vec.util import config_parser
from roadscene2vec.scene_graph.scene_graph import SceneGraph
from roadscene2vec.scene_graph.extraction.image_extractor import RealExtractor
from roadscene2vec.data.dataset import RawImageDataset
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from pathlib import Path
from tqdm import tqdm
from timeit import default_timer as timer

def elapsed_time(func, *args, **kwargs):
  start = timer()
  output = func(*args, **kwargs)
  end = timer()
  print(f'{end - start} seconds elapsed.')
  return output

def elapsed_time(func, *args, **kwargs):
  start = timer()
  output = func(*args, **kwargs)
  end = timer()
  print(f'{end - start} seconds elapsed.')
  return output

# Utilities
def get_extractor(config):
  return RealExtractor(config)

def get_data(extractor):
  temp = RawImageDataset()
  temp.dataset_save_path = extractor.input_path
  return temp.load().data

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

#  frame = frame.transpose(1,2,0) #must do this for cv functionality due to change in real preprocessor
#  img = frame
  plt.subplot(2, 3, 1)
  plt.imshow(cv2_color(frame))
  plt.title("Raw Image")
  plt.axis('off')
  
  bbox_img = draw_bbox(extractor, frame)
  plt.subplot(2, 3, 2)
  plt.imshow(cv2_color(bbox_img))
  plt.title("Object Detection Image")
  plt.axis('off')
  
  #import pdb;pdb.set_trace()
  bev_img = draw_bev(bev, frame)
  plt.subplot(2, 3, 3)
  plt.imshow(cv2_color(bev_img))
  plt.title("Bird's Eye Image")
  plt.axis('off')

  sg_img = draw_scenegraph_pydot(sg)
  plt.subplot(2, 1, 2)
  plt.imshow(sg_img)
  plt.title("SceneGraph Image")
  plt.axis('off')

  # This call is slow!
  if save_path is not None: 
    plt.savefig(save_path, dpi=600)

  plt.show()
  
def visualize(extraction_config):
  extractor = get_extractor(extraction_config)
  dataset_dir = extractor.conf.location_data["input_path"]
  if not os.path.exists(dataset_dir):
      raise FileNotFoundError(dataset_dir)
  all_sequence_dirs = [x for x in Path(dataset_dir).iterdir() if x.is_dir()]
  all_sequence_dirs = sorted(all_sequence_dirs, key=lambda x: int(x.stem.split('_')[0]))  #can't use this lamda to sort in the future
  for path in tqdm(all_sequence_dirs):
      sequence = extractor.load_images(path)
      for frame in sorted(sequence.keys()):
          bbox = get_bbox(extractor, sequence[frame])
          bev = get_bev(extractor)
          sg = get_scenegraph(extractor, bbox, bev)
    
          draw(extractor, sequence[frame], bbox, bev, sg, save_path='output.png')
  print('- finished')
