# roadscene2vec: A Tool for Extracting and Embedding Road Scene-Graphs
## Description
**roadscene2vec** is an open-source tool for extracting and embedding road scene-graphs. 
The goal of **roadscene2vec** is to enable research into the applications and capabilities of road scene-graphs by providing tools for generating scene-graphs and a framework for running graph-learning algorithms on them. We provide tools and utilities for scene graph extraction, dataset processing, graph learning, and results analysis.
For additional details on our library, please reference our research paper located here: 

https://www.sciencedirect.com/science/article/pii/S0950705122000739 (arXiv: https://arxiv.org/abs/2109.01183)

If you find **roadscene2vec** useful for your work please cite our library as follows:

```
@article{malawade2022roadscene2vec,
title = {roadscene2vec: A tool for extracting and embedding road scene-graphs},
journal = {Knowledge-Based Systems},
volume = {242},
pages = {108245},
year = {2022},
issn = {0950-7051},
doi = {https://doi.org/10.1016/j.knosys.2022.108245},
url = {https://www.sciencedirect.com/science/article/pii/S0950705122000739},
author = {Arnav Vaibhav Malawade and Shih-Yuan Yu and Brandon Hsu and Harsimrat Kaeley and Anurag Karra and Mohammad Abdullah {Al Faruque}},
}
```


**Our other related works and associated repositories:**

"_Scene-Graph Augmented Data-Driven Risk Assessment of Autonomous Vehicle Decisions_" (IEEE Transactions on Intellegent Transportation Systems 2021)

https://arxiv.org/abs/2009.06435 

https://github.com/louisccc/sg-risk-assessment


"_Spatio-Temporal Scene-Graph Embedding for Autonomous Vehicle Collision Prediction_" (IEEE IoT Journal 2022)

https://arxiv.org/abs/2111.06123

https://github.com/AICPS/sg-collision-prediction


---
## General Python Setup
All of our experiments were performed on a Linux Server running Ubuntu 16.04 LTS. 
Our library has also been tested on Windows 10 but requires different installation steps for Detectron2 and CARLA.


First, download and install Anaconda here:
https://www.anaconda.com/products/individual

If you are using a GPU, install the corresponding CUDA toolkit for your hardware from Nvidia here:
https://developer.nvidia.com/cuda-toolkit

Next, create a conda virtual environment running Python 3.6:
```shell
conda create --name av python=3.9
```

After setting up your environment. Activate it with the following command:

```shell
conda activate av
```

Install PyTorch to your conda virtual environment by following the instructions here for your CUDA version:
https://pytorch.org/get-started/locally/

In our experiments we used Torch 1.9 and CUDA 10.2 but later versions should also work.


Next, install the PyTorch Geometric library by running the corresponding commands for your Torch and CUDA version:
https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html

Once this setup is completed, install the rest of the requirements from requirements.txt:

```shell
pip install -r requirements.txt
```

If you want to visualize the extracted scene-graphs, as is done in Use-Case 1, you can either use the networkx or pydot/graphviz APIs. Typically graphviz works better so our code defaults to graphviz. In order to render these graphs, you must have [graphviz](https://www.graphviz.org/download/) installed on your system along with the corresponding python package installed as follows:
```
conda install -c anaconda graphviz
```

If you want to use our CARLA dataset generation tools then you need to have the CARLA simulator and Python API installed as described here:
https://github.com/carla-simulator/carla/releases

https://carla.readthedocs.io/en/latest/start_quickstart/


To perform image scene-graph extraction, you must first install Detectron2 by following these instructions:
https://detectron2.readthedocs.io/en/latest/tutorials/install.html

---
## Usage Examples
### Setup and Dataset
Before running any of the use cases, please download the use_case_data directory from the following link and place it into the examples directory.
https://drive.google.com/drive/folders/1zjoixga_S8Ba7khCLBos6nhEhe1UWnv7?usp=sharing

Our raw CARLA dataset can also be downloaded from IEEE Dataport (1043.zip): https://ieee-dataport.org/documents/scenegraph-risk-assessment-dataset


### Use Case 1: Converting an Ego-Centric Observation (Image) into a Scene-Graph
In this use case, we demonstrate how to use roadscene2vec to extract road scenegraphs from a driving clip. In the sample script examples/use_case_1.py, roadscene2vec first takes in the use_case_1_scenegraph_extraction_config.yaml config file. This file specifies the location of the data from which to extract scenegraphs from along with the various relations and actors to include in each scenegraph . A RealScenegraphExtraction object is created using the use_case_1_scenegraph_extraction_config.yaml file. This RealScenegraphExtraction then extracts scenegraphs and saves them as a SceneGraphDataset object. 

To run this use case, cd into the examples folder and run the corresponding module. 

```shell
$ cd examples
$ python use_case_1.py
```

Default specifications are given in the config file, but users can edit the config file used in this use case by heading into the examples folder. .
In use_case_1_scenegraph_extraction_config.yaml the user can specify what type of preprocessed dataset object they would like to extract scenegraphs from - "carla" or "image". 
They can also list the input path of this preprocessed dataset object along with where to save the SceneGraphDataset object created after extraction. Finally, relational settings used during extraction can also be edited in this file. The config file with default arguments along with comments explaining each config parameter is shown below:

```shell
---
dataset_type: 'image' #Type of data from which to extract scenegraphs from. Options: 'image', 'carla'
location_data:
    input_path: '/lanechange' #input path to main directory containing driving sequence subdirectories from which to extract scenegraphs
    data_save_path: '/use_case_1_sg_extraction_output.pkl' #path to save extracted scenegraph dataset

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

NOTE: If you would like to use a new image dataset, we recommend generating a new birds-eye view transformation (bev.json) using the instructions provided here:
https://github.com/AICPS/roadscene2vec/blob/main/roadscene2vec/scene_graph/extraction/bev

### Use Case 2: Using Scene-Graph Embeddings for Subjective Risk Assessment
This use case demonstrates how to use roadscene2vec to classify a given sequence of images as safe or unsafe using risk assessment, which aims to model a driver's subjective analysis of risk on the road. In the sample script examples/use_case_2.py, RealExtractor first extracts a scene graph dataset from the directory of sequences using specifications in the use_case_2_scenegraph_extraction_config.yaml file. Finally, the use_case_2_scenegraph_learning_config.yaml file is used to create a Scenegraph_Trainer object which loads a pre-trained model to output the risk assessment for the created scene graph dataset. 

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
This use case demonstrates how to use roadscene2vec to predict future vehicle collisions using a time-series classification approach which anticipates if collisions will occur in the near future. In the sample script examples/use_case_3.py, RealExtractor first extracts a graph dataset from the directory of sequences using specifications in the use_case_3_scenegraph_extraction_config yaml file. Finally, the use_case_3_scenegraph_learning_config.yaml file is used to create a Scenegraph_Trainer object which loads a pre-trained model to output the collision prediction for the created scene graph dataset. In collision prediction, each frame in a sequence has an output, whereas risk assessment has one output for each sequence. 

To run this use case, use the following commands
```shell
$ cd examples
$ python use_case_3.py
```
This use case uses two configuration objects for each stage of data processing, each of which are derived from their respective yaml files. The paths to the yaml files are arguments used to create the configuration objects.


Data sequences → Scenegraph dataset -  use_case_3_scenegraph_extraction_config.yaml
```shell
scenegraph_extraction_config = configuration(r"use_case_3_scenegraph_extraction_config.yaml",from_function = True)
extracted_scenegraphs = extract_seq(scenegraph_extraction_config)
...
def extract_seq():
sg_extraction_object = RealEx.RealExtractor(scenegraph_extraction_config)
...
#create scenegraph extraction config object
```


Collision Prediction - use_case_3_scenegraph_learning_config.yaml
```shell
training_config = configuration(r"use_case_3_learning_config.yaml",from_function = True)                                                                                                                    
trainer = Scenegraph_Trainer(training_config) #create trainer object using config
trainer.load_model() #load the proper model using the trainer
#model_load_path should contain path to pretrained model in '/pretrained_models/mrgcn_collision_prediction_model.pt'
#load_model should be set to True
```

Arguments provided by these yaml files can again be manipulated by the user.

Note: The only parameters in the learning config that affect this use_case are load_model and model_load_path, as the models are already pretrained. 

Model configuration portion of examples\use_case_3_learning_config.yaml
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
  model_load_path: '/pretrained_models/mrgcn_collision_prediction_model.pt' #path to load pretrained model
  model_save_path: '' #path to save trained model
```

Use_case_3_scenegraph_extraction_config.yaml default values
```shell
dataset_type: 'image' #Type of data from which to extract scenegraphs from. Options: 'image', 'carla'
location_data:
    input_path: '/lanechange' #input path to main directory containing driving sequence subdirectories from which to extract scenegraphs
    data_save_path: '/use_case_3_sg_extraction_output.pkl' #path to save extracted scenegraph dataset

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



### Use Case 4: Evaluating Transfer Learning
In this use case, we demonstrate an example of a transfer learning experiment performed on a model trained using a SceneGraphDataset object and evaluated with another SceneGraphDataset object. 
We first create a Scenegraph_Trainer object using the input_path in use_case_4.yaml config file. We then create the input dataset for the model and train said model. After this we create the transfer learning dataset using the transfer_path in the config file. Finally, we evaluate the model using this entire transfer learning dataset and print the metrics after.

To run this use case, cd into the examples folder and run the corresponding module. 

```shell
$ cd examples
$ python use_case_4.py
```

Default arguments are provided in the use_case_4.yaml config file, but the user can make changes. They can specify the path to the SceneGraphDataset object they would like to train with, along with the path to the SceneGraphDataset object they would like to practice transfer learning with. The user can also specify what type of model they would like to run this experiment on and tune the hyperparameters of said model. The config file with default arguments along with comments explaining each config parameter is shown below:
```shell
---
location_data:
    input_path: '/use_case_4_training_input.pkl'#path to pkl containing scenegraph dataset training/testing data
    transfer_path: '/use_case_4_transfer_learning_input.pkl' #path to transfer dataset for transfer learning
    
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
  model_load_path: '' #path to load pretrained model
  model_save_path: '/use_case_4_trained_model.pt' #path to save trained model
   
training_configuration:
  dataset_type: "scenegraph" #Options: real, scenegraph. scenegraph only option for graph based models
  scenegraph_dataset_type: "real" #type of data extracted to create input scenegraph data. Options: carla, real
  task_type: 'sequence_classification' #Options: sequence_classification, graph_classification, collision_prediction
  n_fold: 5 #number of folds for n-fold cross validation
  learning_rate: 0.0001 #0.00005
  epochs: 200
  split_ratio: 0.3 #train-test split ratio
  downsample: False
  seed: 0 #seed used for train-test split
  batch_size: 32 #batch size of training, testing data
  test_step: 10 #perform test validation every n epochs
  
wandb_configuration:
  entity: ''
  project: ''

```

### Use Case 5: Explainability Analysis
This use case demonstrates how to use roadscene2vec to gain further insight into how a model makes decisions using spatial and temporal attention scores. Given a pre-trained model and a scene graph dataset, the script first creates a Scenegraph Trainer object using the use_case_5_config.yaml. Then, using the trainer’s inference function, the script gathers attention scores for every component of every scene graph. Finally, the parse_attn_weights function visualizes the attention scores in a scene graph for the user. 

To run this use case, use the following commands
```shell
$ cd examples
$ python use_case_5.py
```

Creation of Scenegraph Trainer - use_case_5_config.yaml
```shell
training_config = configuration(r"use_case_5_config.yaml",from_function = True)
…
trainer = Scenegraph_Trainer(training_config, wandb_arg)
trainer.split_dataset()
trainer.load_model()
trainer.loss_func = nn.CrossEntropyLoss()
#model_load_path should contain path to pretrained model
#load_model should be set to True
#set visualize in use_case_5_data to true to visualize the attention scores of the scene graphs
```
Note: The only parameters in the learning config that affect this use_case are load_model and model_load_path, as the models are already pre-trained. 




