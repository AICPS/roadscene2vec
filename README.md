# sg2vec
A Tool for Extracting and Embedding Road Scene-Graphs

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
TODO

### Use Case 3: Using Scene-Graph Embeddings for Collision Prediction
TODO

### Use Case 4: Evaluating Transfer Learning
TODO

### Use Case 5: Explainability Analysis
TODO

