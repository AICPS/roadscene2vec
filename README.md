# sg2vec: A Tool for Extracting and Embedding Road Scene-Graphs
## Description
TODO

---
## General Python Setup

First, download and install Anaconda here:
https://www.anaconda.com/products/individual

If you are using a GPU, install the corresponding CUDA toolkit for your hardware from Nvidia here:
https://developer.nvidia.com/cuda-toolkit

Next, create a conda virtual environment running Python 3.6:
```shell
conda create --name av python=3.6
```

After setting up your environment. Activate it with the following command:

```shell
conda activate av
```

Install PyTorch to your conda virtual environment by following the instructions here for your CUDA version:
https://pytorch.org/get-started/locally/

In our experiments we used Torch 1.5 and 1.6 but later versions should also work fine.


Next, install the PyTorch Geometric library by running the corresponding commands for your Torch and CUDA version:
https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html

Once this setup is completed, install the rest of the requirements from requirements.txt:

```shell
pip install -r requirements.txt
```


---
## Usage Examples
### Use Case 1: Converting an Ego-Centric Observation (Image) into a Scene-Graph
TODO

### Use Case 2: Using Scene-Graph Embeddings for Subjective Risk Assessment
This use case demonstrates how to use SG2VEC to classify a given sequence of images as safe or unsafe using risk assessment, which aims to model a driver's subjective analysis of risk on the road. In the sample script examples/use_case_2.py, RealExtractor first extracts a scene graph dataset from the directory of sequences using specifications in the use_case_2_scenegraph_extraction_config.yaml file. Finally, the use_case_2_scenegraph_learning_config.yaml file is used to create a Scenegraph_Trainer object which loads a pre-trained model to output the risk assessment for the created scene graph dataset. 

To run this use case, use the following commands
```shell
$ cd examples
$ python use_case_2.py
```
This use case uses two configuration objects for each stage of data processing, each of which are derived from their respective yaml files. The paths to the yaml files are arguments used to create the configuration objects.


Data sequences → Scenegraph dataset -  use_case_2_scenegraph_extraction_config.yaml
```shell
scenegraph_extraction_config = configuration(r"use_case_2_scenegraph_extraction_config.yaml",from_function = True)
extracted_scenegraphs = extract_seq(scenegraph_extraction_config)
...
def extract_seq():
sg_extraction_object = RealEx.RealExtractor(scenegraph_extraction_config)
...
#create scenegraph extraction config object
```


Risk assessment -use_case_2_scenegraph_learning_config.yaml
```shell
training_config = configuration(r"use_case_2_learning_config.yaml",from_function = True)                                                                                                                    
trainer = Scenegraph_Trainer(training_config) #create trainer object using config
trainer.load_model() #load the proper model using the trainer
#model_load_path should contain path to pretrained model in '/pretrained_models/mrgcn_sequence_classification_model.pt'
#load_model should be set to True
```
Arguments provided by these yaml files can be manipulated by the user (portion of examples\use_case_2_learning_config.yaml)

Note: The only parameters in the learning config that affect this use_case are load_model and model_load_path, as the models are already pretrained. 

Model configuration portion of examples\use_case_2_learning_config.yaml
```shell
model_configuration:
  num_relations: 12 #num of types of relations extracted, needed only for graph based models 
  model: 'mrgcn' #options: mrgcn, mrgin
  num_layers: 3 #defines number of RGCN conv layers.
  load_model: False #load pretrained model
  num_of_classes: 9 #8 #num of actors
  conv_type: 'FastRGCNConv'
  hidden_dim: 64
  layer_spec: null
  pooling_type: 'sagpool'
  pooling_ratio: 0.5
  readout_type: 'add'
  temporal_type: 'lstm_attn'
  lstm_input_dim: 50
  lstm_output_dim: 20
  nclass: 2 #dimension of final output
  dropout: 0.1 #dropout probability
  device: 'cuda' #device to place training/testing data batches
  activation: 'relu'
  model_load_path: '/pretrained_models/mrgcn_sequence_classification_model.pt' #path to load pretrained model
  model_save_path: '' #path to save trained model
```
Use_case_2_scenegraph_extraction_config.yaml default values
```shell
dataset_type: 'image' #Type of data from which to extract scenegraphs from. Options: 'image', 'carla'
location_data:
    input_path: '/lanechange' #input path to main directory containing driving sequence subdirectories from which to extract scenegraphs
    data_save_path: '/use_case_2_sg_extraction_output.pkl' #path to save extracted scenegraph dataset

relation_extraction_settings:
  frames_limit: null #extract scenegraphs for 1 frame every n frames per sequence subdirectory. currently only functional for image based extraction. Options: null(None), Integer n
  ACTOR_NAMES: ["ego_car", 'car','moto','bicycle','ped','lane','light','sign', 'road'] #types of actors that can be found in data sequences. ego_car actor represents the car from whose perspective we are viewing the road. this array's structure is also used for one-hot encoding when creating node embeddings, "ego_car", "lane", "road" are assumed to always be in this list.
  RELATION_NAMES: ['isIn', 'inDFrontOf', 'inSFrontOf', 'atDRearOf', 'atSRearOf', 'toLeftOf', 'toRightOf', 'near_coll', 'super_near' , 'very_near', 'near' ,'visible'] #types of relations to extract
  
  #actor types specified in proximity_relation_list, directional_relation_list must first be defined in ACTOR_NAMES
  #relations specified in PROXIMITY_THRESHOLDS, DIRECTIONAL_THRESHOLDS, RELATION_COLORS must first be defined in RELATION_NAMES
  PROXIMITY_THRESHOLDS: [['near_coll',4],['super_near',7],['very_near',10],['near',16],['visible',25]] #define proximity relations in the format [relation, distance (ft)] in decreasing order of closeness
  LANE_THRESHOLD: 6 #feet. if object's center is more than this distance away from ego's center, build left or right lane relation. otherwise build middle lane relation
  DIRECTIONAL_THRESHOLDS: [['isIn',[[0,0]]], ['inDFrontOf',[[45,90],[90,135]]], ['inSFrontOf',[[0,45],[135,180]]], ['atDRearOf',[[225,270],[270,315]]], ['atSRearOf',[[180,225],[315,360]]]] #Leftof and Rightof assumed to always be direction relations. additonal directional relations can be specified in the form [[relation], [[1st range of degrees], [2nd range of degrees], ..]]
  RELATION_COLORS: [['isIn','black'],['near_coll','red'], ['super_near','orange'], ['very_near','yellow'], ['near','purple'], ['visible','green'], ['inDFrontOf','violet'], ['inSFrontOf','violet'], ['atDRearOf','turquoise'], ['atSRearOf','turquoise'], ['toLeftOf','blue'], ['toRightOf','blue']] #define relational edge colors for scenegraph visualization purposes in the format [relation, edge_color]
  proximity_relation_list: [['car','road',25], ['ego_car', 'car',25]] #[[ACTORTYPE1, ACTORTYPE2, max proximity distance before relations are not extracted]]
  directional_relation_list: [['car','road',25], ['ego_car', 'car',25]] #[[ACTORTYPE1, ACTORTYPE2, max proximity distance before relations are not extracted]] 
  
  #every type of actor in ACTOR_NAMES can have a list of synonymous names found in the object detection data. for a given ACTOR_NAMES array, all types of objects within the array are treated as objects of type ACTOR.
  MOTO_NAMES: ["moto","Harley-Davidson", "Kawasaki", "Yamaha"]
  BICYCLE_NAMES: ["bicycle","Gazelle", "Diamondback", "Bh"]
  CAR_NAMES: ["car","TRUCK","BUS","Ford", "Bmw", "Toyota", "Nissan", "Mini", "Tesla", "Seat", "Lincoln", "Audi", "Carlamotors", "Citroen", "Mercedes-Benz", "Chevrolet", "Volkswagen", "Jeep", "Nissan", "Dodge", "Mustang"]
  SIGN_NAMES: ["sign"]
  LIGHT_NAMES: ["light"]
  PED_NAMES: []
  ROAD_NAMES: []
  LANE_NAMES: []


image_settings: #path to bev calibration data. only for use with real image scenegraph extraction
    BEV_PATH: '/bev.json'
```




### Use Case 3: Using Scene-Graph Embeddings for Collision Prediction
This use case demonstrates how to use SG2VEC to predict future vehicle collisions using a time-series classification approach which anticipates if collisions will occur in the near future. In the sample script examples/use_case_3.py, RealPreprocessor first creates an image dataset using specifications in the data_config yaml file. The function im2graph then converts the image dataset to a scene graph dataset using specifications in the scenegraph_extraction_config.yaml file. Finally, Scenegraph_Trainer loads a pre-trained model to output the collision prediction for the created scene graph dataset. In collision prediction, each frame in a sequence has an output, whereas risk assessment has one output for each sequence. 

To run this use case, use the following commands
```shell
$ cd examples
$ python use_case_3.py
```
This use case uses three configuration objects for each stage of data processing, each of which are derived from their respective yaml files. The paths to the yaml files are arguments used to create the configuration objects.


Image dataset - data_preprocessing_config.yaml
```shell
def extract_seq():
    real_data_preprocessing_config = configuration(r"use_case_3_data_preprocessing_config.yaml",from_function = True)                                                                                                                             #config should contain path to folder containing sequence to evaluate
```

Scenegraph dataset - scenegraph_extraction_config.yaml
```shell
def im2graph():
    real_scenegraph_extraction_config = configuration(r"use_case_3_scenegraph_extraction_config.yaml",from_function = True)
#config should contain path to pkl containing preprocessed RawImageDataset
```

trainer - learning_config.yaml
```shell
    training_config = configuration(r"use_case_3_learning_config.yaml",from_function = True)
#task_type in learning config training_configuration should be set to collision_prediction
#model_load_path should contain path to pretrained trainer in sg2vec/examples/pretrained_models
#load_model should be set to True
```

Arguments provided by these yaml files can again be manipulated by the user.


### Use Case 4: Evaluating Transfer Learning
TODO

### Use Case 5: Explainability Analysis
TODO

