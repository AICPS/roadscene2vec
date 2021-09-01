import sys, os
sys.path.append(os.path.dirname(sys.path[0]))
from pathlib import Path
import data.carla_preprocessor as cp
import data.real_preprocessor as ip
import scene_graph.extraction.carla_extractor as CarlaEx
import scene_graph.extraction.image_extractor as RealEx
import pickle 
import yaml
import pdb



class configuration():
    def __init__(self,data_config_path):
        with open(data_config_path,"r") as f:    
            self.conf = yaml.load(f)  
            
            
def execute():
    #preprocessing
    #==============================================        
    data_config = configuration(r"C:\users\harsi\av\roadscene2vec\config\data_config_real.yaml")
    if data_config.conf["dataset_type"] == "carla":
        carla_preproc = cp.CarlaPreprocessor(data_config.conf)
        carla_preproc.load()
        base_ds = carla_preproc.getDataSet()
    elif data_config.conf["dataset_type"] == "image":
        img_preproc = ip.RealPreprocessor(data_config.conf)
        img_preproc.load()
        base_ds = img_preproc.getDataSet()
                 
    base_ds.save()
    #=============================================
    
    #extraction
    #=============================================
    scene_config = configuration(r"C:\users\harsi\av\roadscene2vec\config\scene_graph_config_real.yaml")
    if scene_config.conf["dataset_type"] == "carla":
        sg_extraction_object = CarlaEx.CarlaExtractor(scene_config.conf)
        sg_extraction_object.load()
        scenegraph_dataset = sg_extraction_object.getDataSet() #returned scenegraphs from extraction
        scenegraph_dataset.save()
    elif scene_config.conf["dataset_type"] == "image": #must calibrate birds eye view for real data
        sg_extraction_object = RealEx.RealExtractor(scene_config.conf)
        sg_extraction_object.load()
        scenegraph_dataset = sg_extraction_object.getDataSet() #returned scenegraphs from extraction
        scenegraph_dataset.save()
    
#     scenegraph_dataset.scene_graphs[2][17838862].visualize(r"C:\av_data\carla.dot")   
#     dt = scene_config.conf["dataset_type"]
#     for sequence in scenegraph_dataset.scene_graphs:
#         if sequence > 1:
# #             count = 0
#             os.makedirs(f"C:\\av_data\\{dt}\\{sequence}")
#             total_sg = len(scenegraph_dataset.scene_graphs[sequence])
#             for frame in scenegraph_dataset.scene_graphs[sequence]:
# #                 if count == 66:
# #                     pdb.set_trace()
#                 scenegraph_dataset.scene_graphs[sequence][frame].visualize(f"C:\\av_data\\{dt}\\{sequence}\\{frame}.dot") 
#                 count += 1
#             if count != total_sg:
#                 print(f"{sequence} complete visualization failed")
    #     
    scenegraph_dataset.scene_graphs[5569][14490].visualize(r"C:\av_data\real_img.dot")   
#         dot -Tpng c:/av_data/real_img.dot > c:/av_data/real_img.png

    #===============================================
    print("done")
    
    
    #training
    #=========================================================
    #training done here
    
    #=========================================================

if __name__ == "__main__":
    execute()
    
    