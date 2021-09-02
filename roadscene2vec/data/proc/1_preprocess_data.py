import os
import sys

sys.path.append(os.path.dirname(sys.path[0]))
import real_preprocessor as ip
from util.config_parser import configuration

'''This script runs pre-processing of Real data'''
def preprocess_data():
    data_config = configuration(sys.argv[1:])
    if data_config.dataset_type == "image":
        img_preproc = ip.RealPreprocessor(data_config)
        img_preproc.load()
        base_ds = img_preproc.getDataSet()
    else:
        raise ValueError("Please input a valid dataset type")
               
    base_ds.save()
    
if __name__ == "__main__":
    preprocess_data()
    
   
   
   