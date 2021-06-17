import sys, os
sys.path.append(os.path.dirname(sys.path[0]))
import sg2vec.data.real_preprocessor as ip
from sg2vec.util.config_parser import configuration
import sg2vec.scene_graph.extraction.image_extractor as RealEx
from sg2vec.util.trainer import Trainer


def im2graph():
    real_scenegraph_extraction_config = configuration(r"C:\Users\harsi\av\sg2vec\config\scene_graph_config_real.yaml",from_function = True) #replace with path to sg2vec\config\data_config_real.yaml
                                                                                                                                            #config should contain path to pkl containing preprocessed RawImageDataset
    real_scenegraph_extraction_object = RealEx.RealExtractor(real_scenegraph_extraction_config) #creating Real Image Scenegraph Extractor using config
    real_scenegraph_extraction_object.load() #extract scenegraphs for each frame in the sequence using Real Image Scenegraph Extractor
    real_scenegraph_dataset = real_scenegraph_extraction_object.getDataSet() #returned Scenegraph Dataset Object containing the extracted Real Image Scenegraphs for each frame in the sequence 
    return real_scenegraph_dataset

def extract_seq():
    real_data_preprocessing_config = configuration(r"C:\Users\harsi\av\sg2vec\config\data_config_real.yaml",from_function = True) #replace with path to sg2vec\config\scene_graph_config_real.yaml
                                                                                                                                  #config should contain path to folder containing sequence to evaluate
    real_preprocessor = ip.RealPreprocessor(real_data_preprocessing_config) #creating Real Image Preprocessor using config
    real_preprocessor.load() #preprocesses sequence by extracting its frame data
    real_dataset = real_preprocessor.getDataSet() #get RawImageDataset using Real Image Preprocessor
    real_image_scenegraph_dataset = im2graph() #Extract scenegraphs for each frame for the given sequence 
    return real_image_scenegraph_dataset
    
def risk_assess():
    extracted_scenegraphs = extract_seq() #extracted scenegraphs for each frame for the given sequence  
    
    training_config = configuration(r"C:\Users\harsi\av\sg2vec\config\learning_config.yaml",from_function = True) #replace with path to sg2vec\config\learning_config.yaml
                                                                                                                  #task_type in learning config training_configuration should be set to sequence_classification
                                                                                                                  #model_load_path should contain path to pretrained trainer in...
    trainer = Trainer(training_config) #create trainer object using config
    model = trainer.load_model() #load the proper model using the trainer
    model_input = trainer.format_use_case_model_input(extracted_scenegraphs) #get input for the model
    output, _ = self.model.forward(model_input) #output risk assessment for the original sequence 
    return output
    

if __name__ == "__main__":
    print(risk_assess()) #Assess risk of...
    
    #path should be to a folder containing one sequence is that okay? 
    #change comments in risk assess
    
    