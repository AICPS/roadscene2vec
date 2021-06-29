import sys, os
sys.path.append(os.path.dirname(sys.path[0]))
import data.carla_preprocessor as cp
import data.real_preprocessor as ip
from util.config_parser import configuration
from util.script_exceptions import Invalid_Dataset_Type


#python 1_preprocess_dataset.py --yaml_path C:\users\harsi\research\sg2vec\sg2vec\config\data_config_real.yaml  

'''This script runs pre-processing of Carla or Real data'''
def preprocess_data():
    data_config = configuration(sys.argv[1:])
    if data_config.dataset_type == "carla":
        carla_preproc = cp.CarlaPreprocessor(data_config)
        carla_preproc.load()
        base_ds = carla_preproc.getDataSet()
    elif data_config.dataset_type == "image":
        img_preproc = ip.RealPreprocessor(data_config)
        img_preproc.load()
        base_ds = img_preproc.getDataSet()
        print(base_ds.im_height)
        print(base_ds.im_width)
        print(base_ds.color_channels)
        print(base_ds.frame_limit)
    else:
        raise Invalid_Dataset_Type("Please input a valid dataset type")
                 
    base_ds.save()
    
if __name__ == "__main__":
    preprocess_data()
    
   
   
   