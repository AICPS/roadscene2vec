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
This use case demonstrates how to use SG2VEC to classify a given sequence of images as safe or unsafe using risk assessment, which aims to model a driver's subjective analysis of risk on the road. In the sample script examples/use_case_2.py, RealPreprocessor first creates an image dataset using specifications in the data_config yaml file. The function im2graph then converts the image dataset to a scene graph dataset using specifications in the scene_graph_config.yaml file. Finally, Scenegraph_Trainer loads a pre-trained model to output the risk assessment for the created scene graph dataset. 

To run this use case, use the following commands
```shell
$ cd examples
$ python use_case_2.py
```
This use case uses three configuration objects for each stage of data processing, each of which are derived from their respective yaml files. The paths to the yaml files are arguments used to create the configuration objects.


Image dataset - data_preprocessing_config.yaml
```shell
def extract_seq():
    real_data_preprocessing_config = configuration(r"use_case_2_data_preprocessing_config.yaml",from_function = True)
#config should contain path to pkl containing preprocessed RawImageDataset
```

Scenegraph dataset - scenegraph_extraction_config.yaml
```shell
def im2graph():
    real_scenegraph_extraction_config = configuration(r"use_case_2_scenegraph_extraction_config.yaml",from_function = True)
#config should contain path to folder containing sequence to evaluate
```

trainer - learning_config.yaml
```shell
training_config = configuration(r"use_case_2_learning_config.yaml",from_function = True)
#task_type in learning config training_configuration should be set to sequence_classification
#model_load_path should contain path to pretrained trainer in sg2vec/examples/pretrained_models
#load_model should be set to True
```

Arguments provided by these yaml files can be manipulated by the user (portion of examples\use_case_2_learning_config.yaml)

```shell
---
model_configuration:
  model: 'mrgcn' # model type
  num_layers: 3 # number of layers in neural network
  load_model: True # if model to be used is being loaded instead of trained (True for use_case_2.py)
training_configuration:
  task_type: 'sequence_classification' # task to be done
  scenegraph_dataset_type: "real" # carla or real
  n_fold: 1 # number of folds for cross validation
  num_of_classes: 8 # num of actors
  learning_rate: 0.0001 # initial learning rate for the model
  epochs: 1 # number of epochs to train
  model_load_path: 'pretrained_models\20_frame_lanechange_seq_classification_20_epochs.pt' #path to pretrained models
  model_save_path: './model/model_best_val_loss_.vec.pt' # path to save models that are trained
  split_ratio: 0.3 # split ratio for train test split
  downsample: False # Downsampling
  seed: 2063609112 # random seed
  activation: 'relu' # activation function for neural networks
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

