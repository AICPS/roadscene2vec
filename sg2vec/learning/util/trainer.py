import sys, os, pdb
from pathlib import Path
sys.path.append(str(Path("../../")))
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pandas as pd
import random
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn import preprocessing
from matplotlib import pyplot as plt
import math


from sg2vec.data.dataset import SceneGraphDataset
from sg2vec.data.dataset import RawImageDataset
from sg2vec.scene_graph.relation_extractor import Relations
from argparse import ArgumentParser
from tqdm import tqdm

from sg2vec.learning.model.cnn_lstm import CNN_LSTM_Classifier
from sg2vec.learning.model.lstm import LSTM_Classifier
from sg2vec.learning.model.mrgcn import MRGCN
from sg2vec.learning.model.mrgin import MRGIN
from sg2vec.learning.model.cnn import CNN_Classifier
from torch_geometric.data import Data, DataLoader, DataListLoader
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.utils import resample
import pickle as pkl
from sklearn.model_selection import train_test_split, StratifiedKFold
from sg2vec.learning.util.metrics import *

from collections import Counter,defaultdict
import wandb

from sg2vec.learning.util.model_input_preprocessing import *

import sg2vec
sys.modules['data'] = sg2vec.data

class Trainer:

    def __init__(self, config, wandb_a = None):
        self.config = config
        if wandb_a != None:
            self.log = True
            self.wandb = wandb_a
            self.wandb_config = wandb_a.config
            # load config into wandb
            for section in self.config.args:
                for config_arg in self.config.args[section]:
                    self.wandb_config[section+'.'+config_arg] = self.config.args[section][config_arg]
        else:
            self.log = False             
        if self.config.training_configuration["seed"] != None: 
            self.config.seed = self.config.training_configuration["seed"]
            np.random.seed(self.config.seed)
            torch.manual_seed(self.config.seed)
        else:
            self.config.seed = random.randint(0,2**32) #default value
            np.random.seed(self.config.seed)
            torch.manual_seed(self.config.seed)
            
        self.feature_list = set()
        for i in range(self.config.training_configuration['num_of_classes']):
            self.feature_list.add("type_"+str(i))
        self.toGPU = lambda x, dtype: torch.as_tensor(x, dtype=dtype, device=self.config.training_configuration['device'])
        self.best_val_loss = 99999
        self.best_epoch = 0
        self.best_val_acc = 0
        self.best_val_auc = 0
        self.best_val_confusion = []
        self.best_val_f1 = 0
        self.best_val_mcc = -1.0
        self.best_val_acc_balanced = 0
        self.unique_clips = {}
        self.log = False

    #abstract function implemented by subclasses
    def split_dataset(self):
        raise NotImplementedError


    def build_model(self): #this involves changing mrcgn and mrgin files to be compatible with new config tht we pass in
        # BD mode
        self.config.num_features = len(self.feature_list)
        self.config.num_relations = max([r.value for r in Relations])+1
        if self.config.model_configuration["model"] == "mrgcn":
            self.model = MRGCN(self.config).to(self.config.training_configuration["device"])
        elif self.config.model_configuration["model"]  == "mrgin":
            self.model = MRGIN(self.config).to(self.config.training_configuration["device"])
        elif self.config.model_configuration["model"]  == "cnn":
            self.model = CNN_Classifier((self.config.training_configuration['batch_size'], self.image_dataset.frame_limit,self.image_dataset.color_channels, self.image_dataset.im_height, self.image_dataset.im_width), self.config).to(self.config.training_configuration["device"])
        elif self.config.model_configuration["model"]  == "cnn_lstm":
            self.model = CNN_LSTM_Classifier((self.config.training_configuration['batch_size'], self.image_dataset.frame_limit,self.image_dataset.color_channels, self.image_dataset.im_height, self.image_dataset.im_width), self.config).to(self.config.training_configuration["device"])
        elif self.config.model_configuration["model"]  == "lstm":
            self.model = LSTM_Classifier((self.config.training_configuration['batch_size'], self.image_dataset.frame_limit,self.image_dataset.color_channels, self.image_dataset.im_height, self.image_dataset.im_width),'lstm', self.config).to(self.config.training_configuration["device"])        
        elif self.config.model_configuration["model"]  == "gru":
            self.model = LSTM_Classifier((self.config.training_configuration['batch_size'], self.image_dataset.frame_limit,self.image_dataset.color_channels, self.image_dataset.im_height, self.image_dataset.im_width), 'gru', self.config).to(self.config.training_configuration["device"]) 
        else:
            raise Exception("model selection is invalid: " + self.config.model_configuration["model"])
        #
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.training_configuration["learning_rate"], weight_decay=self.config.training_configuration["weight_decay"])
        
        if self.config.model_configuration["load_model"]  == False:
            if self.class_weights.shape[0] < 2:
                self.loss_func = nn.CrossEntropyLoss()
            else:
                self.loss_func = nn.CrossEntropyLoss(weight=self.class_weights.float().to(self.config.training_configuration["device"]))
     
            #wandb.watch(self.model, log="all")
            if self.log:
                self.wandb.watch(self.model, log="all")


    # Pick between Standard Training and KFold Cross Validation Training
    def learn(self):  
        if self.config.training_configuration["n_fold"] <= 1 or self.config.location_data["transfer_path"] != None:
            print('\nRunning Standard Training Loop\n')
            self.train()
        else:
            print('\nRunning {}-Fold Cross Validation Training Loop\n'.format(self.config.training_configuration["n_fold"]))
            self.cross_valid()

    def train(self): #edit
        if (self.config.training_configuration['task_type'] in ['sequence_classification','graph_classification','collision_prediction']):
            tqdm_bar = tqdm(range(self.config.training_configuration['epochs']))
    
            for epoch_idx in tqdm_bar: # iterate through epoch   
                acc_loss_train = 0
                self.sequence_loader = DataListLoader(self.training_data, batch_size=self.config.training_configuration["batch_size"])
    
                for data_list in self.sequence_loader: # iterate through batches of the dataset
                    self.model.train()
                    self.optimizer.zero_grad()
                    labels = torch.empty(0).long().to(self.config.training_configuration["device"])
                    outputs = torch.empty(0,2).to(self.config.training_configuration["device"])
    
                    #need to change below for current implementation
                    for sequence in data_list: # iterate through scene-graph sequences in the batch
                        data, label = sequence['sequence'], sequence['label'] 
                        graph_list = [Data(x=g['node_features'], edge_index=g['edge_index'], edge_attr=g['edge_attr']) for g in data]  
                        self.train_loader = DataLoader(graph_list, batch_size=len(graph_list))
                        sequence = next(iter(self.train_loader)).to(self.config.training_configuration["device"])
                        output, _ = self.model.forward(sequence.x, sequence.edge_index, sequence.edge_attr, sequence.batch)
                        if self.config.training_configuration['task_type'] == 'sequence_classification': # seq vs graph based learning
                            labels  = torch.cat([labels, torch.LongTensor([label]).to(self.config.training_configuration["device"])], dim=0)
                        elif self.config.training_configuration['task_type'] in ['collision_prediction']:
                            label = torch.LongTensor(np.full(output.shape[0], label)).to(self.config.training_configuration["device"]) #fill label to length of the sequence. shape (len_input_sequence, 1)
                            labels  = torch.cat([labels, label], dim=0)
                        else:
                            raise ValueError('task_type is unimplemented')
                        outputs = torch.cat([outputs, output.view(-1, 2)], dim=0) #in this case the output is of shape (len_input_sequence, 2)
                    loss_train = self.loss_func(outputs, labels)
                    loss_train.backward()
                    acc_loss_train += loss_train.detach().cpu().item() * len(data_list)
                    self.optimizer.step()
                    del loss_train
    
                acc_loss_train /= len(self.training_data)
                tqdm_bar.set_description('Epoch: {:04d}, loss_train: {:.4f}'.format(epoch_idx, acc_loss_train))
    
                if epoch_idx % self.config.training_configuration["test_step"] == 0:
                    self.evaluate(epoch_idx)
                    
        else:
            raise ValueError('train(): task type error') 
####

    
    def save_model(self):
        """Function to save the model."""
        saved_path = Path(self.config.training_configuration["model_save_path"]).resolve()
        os.makedirs(os.path.dirname(saved_path), exist_ok=True)
        torch.save(self.model.state_dict(), str(saved_path))
        with open(os.path.dirname(saved_path) + "/model_parameters.txt", "w+") as f:
            f.write(str(self.config))
            f.write('\n')
            f.write(str(' '.join(sys.argv)))

    def load_model(self):
        """Function to load the model."""
        saved_path = Path(self.config.training_configuration["model_load_path"]).resolve()
        if saved_path.exists():
            self.build_model()
            self.model.load_state_dict(torch.load(str(saved_path)))
            self.model.eval()
            



           
            


class Image_Trainer(Trainer):
    def __init__(self, config, wandb_a = None):
        super(Image_Trainer, self).__init__(config, wandb_a)


    def split_dataset(self): #this is init_dataset from multimodal
        if self.config.training_configuration['task_type'] == 'cnn image classification':
            
            self.training_data, self.testing_data, self.feature_list = self.build_real_image_dataset()
            self.training_labels = np.array([i[1] for i in self.training_data])
            self.testing_labels = np.array([i[1] for i in self.testing_data])
            self.training_clip_name = np.array([i[2] for i in self.training_data])
            self.testing_clip_name = np.array([i[2] for i in self.testing_data])
            self.training_data = np.stack([i[0] for i in self.training_data], axis=0)
            self.testing_data = np.stack([i[0] for i in self.testing_data], axis=0)
            self.class_weights = torch.from_numpy(compute_class_weight('balanced', np.unique(self.training_labels), self.training_labels))
            
            if self.config.training_configuration["n_fold"] <= 1:
                print("Number of Training Sequences Included: ", len(self.training_data))
                print("Number of Testing Sequences Included: ", len(self.testing_data))
                print("Num of Training Labels in Each Class: " + str(np.unique(self.training_labels, return_counts=True)[1]) + ", Class Weights: " + str(self.class_weights))
                print("Num of Testing Labels in Each Class: " + str(np.unique(self.testing_labels, return_counts=True)[1]) + ", Class Weights: " + str(self.class_weights)) 
        else:
            raise ValueError('split_dataset(): task type error') 
        

    '''Returns lists of tuples train and test each containing (data, label, category)'''
    def build_real_image_dataset(self):
        image_dataset = RawImageDataset()
        image_dataset.dataset_save_path = self.config.location_data["input_path"]
        self.image_dataset = image_dataset.load()
        self.feature_list = set()
        for i in range(self.config.training_configuration['num_of_classes']):
            self.feature_list.add("type_"+str(i))
              
        class_0 = []
        class_1 = []
        class_0_clip_name = []
        class_1_clip_name = []
          
        for seq in tqdm(self.image_dataset.labels): # for each seq (num total seq,frame,chan,h,w)
            category = self.image_dataset.action_types[seq]
            if category in self.unique_clips:
                self.unique_clips[category] += 1
            else:
                self.unique_clips[category] = 1
            seq_data = np.stack([value for value in self.image_dataset.data[seq].values()], axis=0)
            if self.image_dataset.labels[seq] == 0:
                class_0.append((seq_data,0,category))                                                  
            elif self.image_dataset.labels[seq] == 1:
                class_1.append((seq_data,1,category))
        y_0 = [0]*len(class_0)  
        y_1 = [1]*len(class_1)
        min_number = min(len(class_0), len(class_1))
        
        if self.config.training_configuration['downsample']:
            modified_class_0, modified_y_0 = resample(class_0, y_0, n_samples=min_number)
        else:
            modified_class_0, modified_y_0 = class_0, y_0
        train, test, train_y, test_y = train_test_split(modified_class_0+class_1, modified_y_0+y_1, test_size=self.config.training_configuration['split_ratio'], shuffle=True, stratify=modified_y_0+y_1, random_state=self.config.seed)
        if self.config.location_data["transfer_path"] != None:
            test, _ = pkl.load(open(self.config.location_data["transfer_path"], "rb"))
            image_sequence = class_1+class_0
            return image_sequence, test, self.feature_list 
        #dont do kfold here instead it is done when learn() is called
        return train, test, self.feature_list # redundant return of self.feature_list #TODO: address this issue


    def train(self):
        if self.config.training_configuration['task_type'] == 'cnn image classification':
            tqdm_bar = tqdm(range(self.config.training_configuration['epochs']))
            for epoch_idx in tqdm_bar: # iterate through epoch   
                acc_loss_train = 0
                permutation = np.random.permutation(len(self.training_data)) # shuffle dataset before each epoch
                self.model.train()
                for i in range(0, len(self.training_data), self.config.training_configuration['batch_size']): # iterate through batches of the dataset
                    batch_index = i + self.config.training_configuration['batch_size'] if i + self.config.training_configuration['batch_size'] <= len(self.training_data) else len(self.training_data)
                    indices = permutation[i:batch_index]
                    batch_x, batch_y = self.training_data[indices], self.training_labels[indices]
                    batch_x, batch_y = self.toGPU(batch_x, torch.float32), self.toGPU(batch_y, torch.long)
                    output = self.model.forward(batch_x).view(-1, 2)
                    loss_train = self.loss_func(output, batch_y)
                    loss_train.backward()
                    acc_loss_train += loss_train.detach().cpu().item() * len(indices)
                    self.optimizer.step()
                    del loss_train
    
                acc_loss_train /= len(self.training_data)
                tqdm_bar.set_description('Epoch: {:04d}, loss_train: {:.4f}'.format(epoch_idx, acc_loss_train))
                
                # no cross validation 
                if epoch_idx % self.config.training_configuration['test_step'] == 0:
                    self.eval_model(epoch_idx)                   
        else:
            raise ValueError
              
    def model_inference(self, X, y, clip_name):
        labels = torch.LongTensor().to(self.config.training_configuration['device'])
        outputs = torch.FloatTensor().to(self.config.training_configuration['device'])
        # Dictionary storing (output, label) pair for all driving categories
        categories = {'outputs': outputs, 'labels': labels}
        batch_size = self.config.training_configuration['batch_size'] # NOTE: set to 1 when profiling or calculating inference time.
        acc_loss = 0
        inference_time = 0
        prof_result = ""
    
        with torch.autograd.profiler.profile(enabled=False, use_cuda=True) as prof:
            with torch.no_grad():
                self.model.eval()
    
                for i in range(0, len(X), batch_size): # iterate through subsequences
                    batch_index = i + batch_size if i + batch_size <= len(X) else len(X)
                    batch_x, batch_y, batch_clip_name = X[i:batch_index], y[i:batch_index], clip_name[i:batch_index]
                    batch_x, batch_y = self.toGPU(batch_x, torch.float32), self.toGPU(batch_y, torch.long)
                    #start = torch.cuda.Event(enable_timing=True)
                    #end =  torch.cuda.Event(enable_timing=True)
                    #start.record()
                    output = self.model.forward(batch_x).view(-1, 2)
                    #end.record()
                    #torch.cuda.synchronize()
                    inference_time += 0#start.elapsed_time(end)
                    loss_test = self.loss_func(output, batch_y)
                    acc_loss += loss_test.detach().cpu().item() * len(batch_y)
                    # store output, label statistics
                    self.update_categorical_outputs(categories, output, batch_y, batch_clip_name)
        # calculate one risk score per sequence (this is not implemented for each category)
        sum_seq_len = 0
        num_risky_sequences = 0
        num_safe_sequences = 0
        correct_risky_seq = 0
        correct_safe_seq = 0
        incorrect_risky_seq = 0
        incorrect_safe_seq = 0
        sequences = len(categories['labels'])
        for indices in range(sequences):
            seq_output = categories['outputs'][indices]
            label = categories['labels'][indices]
            pred = torch.argmax(seq_output)
              
            # risky clip
            if label == 1:
                num_risky_sequences += 1
                sum_seq_len += seq_output.shape[0]
                correct_risky_seq += self.correctness(label, pred)
                incorrect_risky_seq += self.correctness(label, pred)
            # non-risky clip
            elif label == 0:
                num_safe_sequences += 1
                incorrect_safe_seq += self.correctness(label, pred)
                correct_safe_seq += self.correctness(label, pred)
          
        avg_risky_seq_len = sum_seq_len / num_risky_sequences # sequence length for comparison with the prediction frame metric. 
        seq_tpr = correct_risky_seq / num_risky_sequences
        seq_fpr = incorrect_safe_seq / num_safe_sequences
        seq_tnr = correct_safe_seq / num_safe_sequences
        seq_fnr = incorrect_risky_seq / num_risky_sequences
        if prof != None:
            prof_result = prof.key_averages().table(sort_by="cuda_time_total")
    
        return  categories, \
                  acc_loss/len(X), \
                  avg_risky_seq_len, \
                  inference_time, \
                  prof_result, \
                  seq_tpr, \
                  seq_fpr, \
                  seq_tnr, \
                  seq_fnr

    def correctness(self, output, pred):
        return 1 if output == pred else 0

    def update_categorical_outputs(self, categories, outputs, labels, clip_name):
        '''
            Aggregates output, label pairs for every driving category
        '''
        n = len(clip_name)
        for i in range(n):

            categories['outputs'] = torch.cat([categories['outputs'], torch.unsqueeze(outputs[i], dim=0)], dim=0)
            categories['labels'] = torch.cat([categories['labels'], torch.unsqueeze(labels[i], dim=0)], dim=0)

        # reshape outputs

        categories['outputs'] = categories['outputs'].reshape(-1, 2)
    
    def eval_model(self, current_epoch=None):
        metrics = {}
        categories_train, \
        acc_loss_train, \
        train_avg_seq_len, \
        train_inference_time, \
        train_profiler_result, \
        seq_tpr, seq_fpr, seq_tnr, seq_fnr = self.model_inference(self.training_data, self.training_labels, self.training_clip_name) 
    
        # Collect metrics from all driving categories
        metrics['train'] = get_metrics(categories_train['outputs'], categories_train['labels'])
        metrics['train']['loss'] = acc_loss_train
        metrics['train']['avg_seq_len'] = train_avg_seq_len
        metrics['train']['seq_tpr'] = seq_tpr
        metrics['train']['seq_tnr'] = seq_tnr
        metrics['train']['seq_fpr'] = seq_fpr
        metrics['train']['seq_fnr'] = seq_fnr


        categories_test, \
        acc_loss_test, \
        val_avg_seq_len, \
        test_inference_time, \
        test_profiler_result, \
        seq_tpr, seq_fpr, seq_tnr, seq_fnr = self.model_inference(self.testing_data, self.testing_labels, self.testing_clip_name) 
    
        # Collect metrics from all driving categories
        metrics['test'] = get_metrics(categories_test['outputs'], categories_test['labels'])
        metrics['test']['loss'] = acc_loss_test
        metrics['test']['avg_seq_len'] = val_avg_seq_len
        metrics['test']['seq_tpr'] = seq_tpr
        metrics['test']['seq_tnr'] = seq_tnr
        metrics['test']['seq_fpr'] = seq_fpr
        metrics['test']['seq_fnr'] = seq_fnr
        metrics['avg_inf_time'] = (train_inference_time + test_inference_time) / ((len(self.training_labels) + len(self.testing_labels))*5)

    
           
        print("\ntrain loss: " + str(acc_loss_train) + ", acc:", metrics['train']['acc'], metrics['train']['confusion'], "mcc:", metrics['train']['mcc'], \
                 "\ntest loss: " +  str(acc_loss_test) + ", acc:",  metrics['test']['acc'],  metrics['test']['confusion'], "mcc:", metrics['test']['mcc'])
    
        self.update_im_best_metrics(metrics, current_epoch)
        metrics['best_epoch'] = self.best_epoch
        metrics['best_val_loss'] = self.best_val_loss
        metrics['best_val_acc'] = self.best_val_acc
        metrics['best_val_auc'] = self.best_val_auc
        metrics['best_val_conf'] = self.best_val_confusion
        metrics['best_val_f1'] = self.best_val_f1
        metrics['best_val_mcc'] = self.best_val_mcc
        metrics['best_val_acc_balanced'] = self.best_val_acc_balanced
           
        if self.config.training_configuration['n_fold'] <= 1 and self.log:  
            self.log2wandb(metrics)
    
        return categories_train, categories_test, metrics
   
    def update_im_best_metrics(self, metrics, current_epoch):
        if metrics['test']['loss'] < self.best_val_loss:
            self.best_val_loss = metrics['test']['loss']
            self.best_epoch = current_epoch if current_epoch != None else self.config.epochs
            self.best_val_acc = metrics['test']['acc']
            self.best_val_auc = metrics['test']['auc']
            self.best_val_confusion = metrics['test']['confusion']
            self.best_val_f1 = metrics['test']['f1']
            self.best_val_mcc = metrics['test']['mcc']
            self.best_val_acc_balanced = metrics['test']['balanced_acc']
            #self.save_model()
                    
    def update_im_cross_valid_metrics(self, categories_train, categories_test, metrics):
        '''
            Stores cross-validation metrics for all driving categories
        '''
        datasets = ['train', 'test']
        if self.fold == 1:
            for dataset in datasets:
                categories = categories_train if dataset == 'train' else categories_test
                self.results['outputs'+'_'+dataset] = categories['outputs']
                self.results['labels'+'_'+dataset] = categories['labels']
                self.results[dataset] = metrics[dataset]
                self.results[dataset]['loss'] = metrics[dataset]['loss']
                self.results[dataset]['avg_seq_len']  = metrics[dataset]['avg_seq_len'] 
                
                # Best results
                self.results['avg_inf_time']  = metrics['avg_inf_time']
                self.results['best_epoch']    = metrics['best_epoch']
                self.results['best_val_loss'] = metrics['best_val_loss']
                self.results['best_val_acc']  = metrics['best_val_acc']
                self.results['best_val_auc']  = metrics['best_val_auc']
                self.results['best_val_conf'] = metrics['best_val_conf']
                self.results['best_val_f1']   = metrics['best_val_f1']
                self.results['best_val_mcc']  = metrics['best_val_mcc']
                self.results['best_val_acc_balanced'] = metrics['best_val_acc_balanced']

    
        else:
            for dataset in datasets:
                categories = categories_train if dataset == 'train' else categories_test
                self.results['outputs'+'_'+dataset] = torch.cat((self.results['outputs'+'_'+dataset], categories['outputs']), dim=0)
                self.results['labels'+'_'+dataset]  = torch.cat((self.results['labels'+'_'+dataset], categories['labels']), dim=0)
                self.results[dataset]['loss'] = np.append(self.results[dataset]['loss'], metrics[dataset]['loss'])
                self.results[dataset]['avg_seq_len']  = np.append(self.results[dataset]['avg_seq_len'], metrics[dataset]['avg_seq_len'])
                
                # Best results
                self.results['avg_inf_time']  = np.append(self.results['avg_inf_time'], metrics['avg_inf_time'])
                self.results['best_epoch']    = np.append(self.results['best_epoch'], metrics['best_epoch'])
                self.results['best_val_loss'] = np.append(self.results['best_val_loss'], metrics['best_val_loss'])
                self.results['best_val_acc']  = np.append(self.results['best_val_acc'], metrics['best_val_acc'])
                self.results['best_val_auc']  = np.append(self.results['best_val_auc'], metrics['best_val_auc'])
                self.results['best_val_conf'] = np.append(self.results['best_val_conf'], metrics['best_val_conf'])
                self.results['best_val_f1']   = np.append(self.results['best_val_f1'], metrics['best_val_f1'])
                self.results['best_val_mcc']  = np.append(self.results['best_val_mcc'], metrics['best_val_mcc'])
                self.results['best_val_acc_balanced'] = np.append(self.results['best_val_acc_balanced'], metrics['best_val_acc_balanced'])

            
        # Log final averaged results
        if self.fold == self.config.training_configuration['n_fold']:
            final_metrics = {}
            for dataset in datasets:
                final_metrics[dataset] = get_metrics(self.results['outputs'+'_'+dataset], self.results['labels'+'_'+dataset])
                final_metrics[dataset]['loss'] = np.average(self.results[dataset]['loss'])
                final_metrics[dataset]['avg_seq_len'] = np.average(self.results[dataset]['avg_seq_len'])

                # Best results
                final_metrics['avg_inf_time']  = np.average(self.results['avg_inf_time'])
                final_metrics['best_epoch']    = np.average(self.results['best_epoch'])
                final_metrics['best_val_loss'] = np.average(self.results['best_val_loss'])
                final_metrics['best_val_acc']  = np.average(self.results['best_val_acc'])
                final_metrics['best_val_auc']  = np.average(self.results['best_val_auc'])
                final_metrics['best_val_conf'] = self.results['best_val_conf']
                final_metrics['best_val_f1']   = np.average(self.results['best_val_f1'])
                final_metrics['best_val_mcc']  = np.average(self.results['best_val_mcc'])
                final_metrics['best_val_acc_balanced'] = np.average(self.results['best_val_acc_balanced'])
    
            print('\nFinal Averaged Results')
            print("\naverage train loss: " + str(final_metrics['train']['loss']) + ", average acc:", final_metrics['train']['acc'], final_metrics['train']['confusion'], final_metrics['train']['auc'], \
                "\naverage test loss: " +  str(final_metrics['test']['loss']) + ", average acc:", final_metrics['test']['acc'],  final_metrics['test']['confusion'], final_metrics['test']['auc'])
            if self.log:
                self.log2wandb(final_metrics)
            
            # final combined results and metrics
            return self.results['outputs_train'], self.results['labels_train'], self.results['outputs_test'], self.results['labels_test'], final_metrics

    def cross_valid(self):
        # KFold cross validation with similar class distribution in each fold
        skf = StratifiedKFold(n_splits=self.config.training_configuration["n_fold"])
        X = np.append(self.training_data, self.testing_data, axis=0)
        clip_name = np.append(self.training_clip_name, self.testing_clip_name, axis=0)
        y = np.append(self.training_labels, self.testing_labels, axis=0)

        # self.results stores average metrics for the the n_folds
        self.results = {}
        self.fold = 1

        # Split training and testing data based on n_splits (Folds)
        for train_index, test_index in skf.split(X, y):
            self.training_data, self.testing_data, self.training_clip_name, self.testing_clip_name, self.training_labels, self.testing_labels = None, None, None, None, None, None #clear vars to save memory
            X_train, X_test = X[train_index], X[test_index]
            clip_train, clip_test = clip_name[train_index], clip_name[test_index]
            y_train, y_test = y[train_index], y[test_index]
            self.class_weights = torch.from_numpy(compute_class_weight('balanced', np.unique(y_train), y_train))

            # Update dataset
            self.training_data = X_train
            self.testing_data  = X_test
            self.training_clip_name = clip_train
            self.testing_clip_name = clip_test
            self.training_labels = y_train
            self.testing_labels  = y_test

            print('\nFold {}'.format(self.fold))
            print("Number of Training Sequences Included: ", len(X_train))
            print("Number of Testing Sequences Included: ",  len(X_test))
            print("Num of Training Labels in Each Class: " + str(np.unique(self.training_labels, return_counts=True)[1]) + ", Class Weights: " + str(self.class_weights))
            print("Num of Testing Labels in Each Class: "  + str(np.unique(self.testing_labels, return_counts=True)[1]) + ", Class Weights: " + str(self.class_weights))
           
            self.best_val_loss = 99999
            self.train()
            #self.log = True
            categories_train, categories_test, metrics = self.eval_model(self.fold)
            self.update_im_cross_valid_metrics(categories_train, categories_test, metrics)
            #self.log = False

            if self.fold != self.config.training_configuration["n_fold"]:
                self.reset_weights(self.model)
                del self.optimizer
                self.build_model()
                
            self.fold += 1            
        del self.results
        
    def reset_weights(self, model):
        for layer in model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def update_cross_valid_metrics(self, categories_train, categories_test, metrics):
            '''
                Stores cross-validation metrics for all driving categories
            '''
            datasets = ['train', 'test']
            if self.fold == 1:
                for dataset in datasets:
                    categories = categories_train if dataset == 'train' else categories_test
                    for category in self.unique_clips.keys():
                        if category == 'all':
                            self.results['outputs'+'_'+dataset] = categories['all']['outputs']
                            self.results['labels'+'_'+dataset] = categories['all']['labels']
                            self.results[dataset] = metrics[dataset]
                            self.results[dataset]['loss'] = metrics[dataset]['loss']
                            self.results[dataset]['avg_seq_len']  = metrics[dataset]['avg_seq_len'] 
                            
                            # Best results
                            self.results['avg_inf_time']  = metrics['avg_inf_time']
                            self.results['best_epoch']    = metrics['best_epoch']
                            self.results['best_val_loss'] = metrics['best_val_loss']
                            self.results['best_val_acc']  = metrics['best_val_acc']
                            self.results['best_val_auc']  = metrics['best_val_auc']
                            self.results['best_val_conf'] = metrics['best_val_conf']
                            self.results['best_val_f1']   = metrics['best_val_f1']
                            self.results['best_val_mcc']  = metrics['best_val_mcc']
                            self.results['best_val_acc_balanced'] = metrics['best_val_acc_balanced']
                        elif category == "lanechange":
                            self.results['outputs'+'_'+dataset] = categories['all']['outputs']
                            self.results['labels'+'_'+dataset] = categories['all']['labels']
                            self.results[dataset] = metrics[dataset]
                            self.results[dataset]['loss'] = metrics[dataset]['loss']
                            self.results[dataset]['avg_seq_len']  = metrics[dataset]['avg_seq_len'] 
                            
                            # Best results
                            self.results['avg_inf_time']  = metrics['avg_inf_time']
                            self.results['best_epoch']    = metrics['best_epoch']
                            self.results['best_val_loss'] = metrics['best_val_loss']
                            self.results['best_val_acc']  = metrics['best_val_acc']
                            self.results['best_val_auc']  = metrics['best_val_auc']
                            self.results['best_val_conf'] = metrics['best_val_conf']
                            self.results['best_val_f1']   = metrics['best_val_f1']
                            self.results['best_val_mcc']  = metrics['best_val_mcc']
                            self.results['best_val_acc_balanced'] = metrics['best_val_acc_balanced']                            
                        else:
                            self.results[dataset][category]['outputs'] = categories[category]['outputs']
                            self.results[dataset][category]['labels'] = categories[category]['labels']
    
            else:
                for dataset in datasets:
                    categories = categories_train if dataset == 'train' else categories_test
                    for category in self.unique_clips.keys():
                        if category == 'all':
                            self.results['outputs'+'_'+dataset] = torch.cat((self.results['outputs'+'_'+dataset], categories['all']['outputs']), dim=0)
                            self.results['labels'+'_'+dataset]  = torch.cat((self.results['labels'+'_'+dataset], categories['all']['labels']), dim=0)
                            self.results[dataset]['loss'] = np.append(self.results[dataset]['loss'], metrics[dataset]['loss'])
                            self.results[dataset]['avg_seq_len']  = np.append(self.results[dataset]['avg_seq_len'], metrics[dataset]['avg_seq_len'])
                            
                            # Best results
                            self.results['avg_inf_time']  = np.append(self.results['avg_inf_time'], metrics['avg_inf_time'])
                            self.results['best_epoch']    = np.append(self.results['best_epoch'], metrics['best_epoch'])
                            self.results['best_val_loss'] = np.append(self.results['best_val_loss'], metrics['best_val_loss'])
                            self.results['best_val_acc']  = np.append(self.results['best_val_acc'], metrics['best_val_acc'])
                            self.results['best_val_auc']  = np.append(self.results['best_val_auc'], metrics['best_val_auc'])
                            self.results['best_val_conf'] = np.append(self.results['best_val_conf'], metrics['best_val_conf'])
                            self.results['best_val_f1']   = np.append(self.results['best_val_f1'], metrics['best_val_f1'])
                            self.results['best_val_mcc']  = np.append(self.results['best_val_mcc'], metrics['best_val_mcc'])
                            self.results['best_val_acc_balanced'] = np.append(self.results['best_val_acc_balanced'], metrics['best_val_acc_balanced'])
                        elif category == "lanechange":
                            self.results['outputs'+'_'+dataset] = torch.cat((self.results['outputs'+'_'+dataset], categories['all']['outputs']), dim=0)
                            self.results['labels'+'_'+dataset]  = torch.cat((self.results['labels'+'_'+dataset], categories['all']['labels']), dim=0)
                            self.results[dataset]['loss'] = np.append(self.results[dataset]['loss'], metrics[dataset]['loss'])
                            self.results[dataset]['avg_seq_len']  = np.append(self.results[dataset]['avg_seq_len'], metrics[dataset]['avg_seq_len'])
                            
                            # Best results
                            self.results['avg_inf_time']  = np.append(self.results['avg_inf_time'], metrics['avg_inf_time'])
                            self.results['best_epoch']    = np.append(self.results['best_epoch'], metrics['best_epoch'])
                            self.results['best_val_loss'] = np.append(self.results['best_val_loss'], metrics['best_val_loss'])
                            self.results['best_val_acc']  = np.append(self.results['best_val_acc'], metrics['best_val_acc'])
                            self.results['best_val_auc']  = np.append(self.results['best_val_auc'], metrics['best_val_auc'])
                            self.results['best_val_conf'] = np.append(self.results['best_val_conf'], metrics['best_val_conf'])
                            self.results['best_val_f1']   = np.append(self.results['best_val_f1'], metrics['best_val_f1'])
                            self.results['best_val_mcc']  = np.append(self.results['best_val_mcc'], metrics['best_val_mcc'])
                            self.results['best_val_acc_balanced'] = np.append(self.results['best_val_acc_balanced'], metrics['best_val_acc_balanced'])
                        else:
                            self.results[dataset][category]['outputs'] = torch.cat((self.results[dataset][category]['outputs'], categories[category]['outputs']), dim=0)
                            self.results[dataset][category]['labels']  = torch.cat((self.results[dataset][category]['labels'], categories[category]['labels']), dim=0)
                
            # Log final averaged results
            if self.fold ==  self.config.training_configuration["n_fold"]:
                final_metrics = {}
                for dataset in datasets:
                    for category in self.unique_clips.keys():
                        if category == 'all':
                            final_metrics[dataset] = get_metrics(self.results['outputs'+'_'+dataset], self.results['labels'+'_'+dataset])
                            final_metrics[dataset]['loss'] = np.average(self.results[dataset]['loss'])
                            final_metrics[dataset]['avg_seq_len'] = np.average(self.results[dataset]['avg_seq_len'])
    
                            # Best results
                            final_metrics['avg_inf_time']  = np.average(self.results['avg_inf_time'])
                            final_metrics['best_epoch']    = np.average(self.results['best_epoch'])
                            final_metrics['best_val_loss'] = np.average(self.results['best_val_loss'])
                            final_metrics['best_val_acc']  = np.average(self.results['best_val_acc'])
                            final_metrics['best_val_auc']  = np.average(self.results['best_val_auc'])
                            final_metrics['best_val_conf'] = self.results['best_val_conf']
                            final_metrics['best_val_f1']   = np.average(self.results['best_val_f1'])
                            final_metrics['best_val_mcc']  = np.average(self.results['best_val_mcc'])
                            final_metrics['best_val_acc_balanced'] = np.average(self.results['best_val_acc_balanced'])
                        elif category == 'lanechange':
                            final_metrics[dataset] = get_metrics(self.results['outputs'+'_'+dataset], self.results['labels'+'_'+dataset])
                            final_metrics[dataset]['loss'] = np.average(self.results[dataset]['loss'])
                            final_metrics[dataset]['avg_seq_len'] = np.average(self.results[dataset]['avg_seq_len'])
    
                            # Best results
                            final_metrics['avg_inf_time']  = np.average(self.results['avg_inf_time'])
                            final_metrics['best_epoch']    = np.average(self.results['best_epoch'])
                            final_metrics['best_val_loss'] = np.average(self.results['best_val_loss'])
                            final_metrics['best_val_acc']  = np.average(self.results['best_val_acc'])
                            final_metrics['best_val_auc']  = np.average(self.results['best_val_auc'])
                            final_metrics['best_val_conf'] = self.results['best_val_conf']
                            final_metrics['best_val_f1']   = np.average(self.results['best_val_f1'])
                            final_metrics['best_val_mcc']  = np.average(self.results['best_val_mcc'])
                            final_metrics['best_val_acc_balanced'] = np.average(self.results['best_val_acc_balanced'])
                        else: 
                            final_metrics[dataset][category] = get_metrics(self.results[dataset][category]['outputs'], self.results[dataset][category]['labels'])
    
                print('\nFinal Averaged Results')
                print("\naverage train loss: " + str(final_metrics['train']['loss']) + ", average acc:", final_metrics['train']['acc'], final_metrics['train']['confusion'], final_metrics['train']['auc'], \
                    "\naverage test loss: " +  str(final_metrics['test']['loss']) + ", average acc:", final_metrics['test']['acc'],  final_metrics['test']['confusion'], final_metrics['test']['auc'])
                if self.log:
                    self.log2wandb(final_metrics)
                
                # final combined results and metrics
                return self.results['outputs_train'], self.results['labels_train'], self.results['outputs_test'], self.results['labels_test'], final_metrics


        
    def log2wandb(self, metrics):
        '''
            Log metrics from all driving categories
        '''
        for category in self.unique_clips.keys():
            if category == 'all':
                log_im_wandb(metrics)
            elif category == 'lanechange':
                log_im_wandb(metrics)
            else:
                log_wandb_categories(metrics, id=category)