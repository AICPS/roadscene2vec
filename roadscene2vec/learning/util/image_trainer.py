import sys
from pathlib import Path
sys.path.append(str(Path("../../")))
import torch
import numpy as np
from tqdm import tqdm
from roadscene2vec.learning.util.trainer import Trainer
from roadscene2vec.data.dataset import RawImageDataset
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import shuffle
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, StratifiedKFold
from roadscene2vec.learning.util.metrics import get_metrics, log_im_wandb, log_wandb_categories

'''Class implementing image based model training including support for splitting input dataset, cross-validation functionality, model inference metrics, and model evaluation.'''
class Image_Trainer(Trainer):
    def __init__(self, config, wandb_a = None):
        '''Class object initialization requires Config Parser object.'''
        super(Image_Trainer, self).__init__(config, wandb_a)

    def split_dataset(self): #this is init_dataset from multimodal
        if (self.config.training_configuration['task_type'] in ['sequence_classification','collision_prediction']):
            self.training_data, self.testing_data = self.build_real_image_dataset()
            self.training_labels = np.array([i[1] for i in self.training_data])
            self.testing_labels = np.array([i[1] for i in self.testing_data])
            self.total_train_labels = np.concatenate([np.full(len(i[0]), i[1]) for i in self.training_data]) # used to compute frame-level class weighting
            self.total_test_labels  = np.concatenate([np.full(len(i[0]), i[1]) for i in self.testing_data])
            self.training_clip_name = np.array([i[2] for i in self.training_data])
            self.testing_clip_name = np.array([i[2] for i in self.testing_data])
            self.training_data = np.stack([i[0] for i in self.training_data], axis=0) #resulting shape is (sequence, image, channel, height, width)
            self.testing_data = np.stack([i[0] for i in self.testing_data], axis=0)
            if self.config.training_configuration['task_type'] == "sequence_classification":
              self.class_weights = torch.from_numpy(compute_class_weight('balanced', 
                                                                         classes=np.unique(self.training_labels), 
                                                                         y=self.training_labels))
              if self.config.training_configuration["n_fold"] <= 1:
                  print("Number of Training Sequences Included: ", len(self.training_data))
                  print("Number of Testing Sequences Included: ", len(self.testing_data))
                  print("Num of Training Labels in Each Class: " + str(np.unique(self.training_labels, return_counts=True)[1]) + ", Class Weights: " + str(self.class_weights))
                  print("Num of Testing Labels in Each Class: " + str(np.unique(self.testing_labels, return_counts=True)[1]) + ", Class Weights: " + str(self.class_weights)) 
            elif self.config.training_configuration['task_type'] == "collision_prediction":
                self.class_weights = torch.from_numpy(compute_class_weight('balanced', 
                                                                           classes=np.unique(self.total_train_labels), 
                                                                           y=self.total_train_labels))
                if self.config.training_configuration["n_fold"] <= 1:
                    print("Number of Training Sequences Included: ", len(self.training_data))
                    print("Number of Testing Sequences Included: ", len(self.testing_data))
                    print("Number of Training Labels in Each Class: " + str(np.unique(self.total_train_labels, return_counts=True)[1]) + ", Class Weights: " + str(self.class_weights))
                    print("Number of Testing Labels in Each Class: " + str(np.unique(self.total_test_labels, return_counts=True)[1]) + ", Class Weights: " + str(self.class_weights))
        else:
            raise ValueError('split_dataset(): task type error') 
        

    def prep_dataset(self, image_dataset):
        class_0 = []
        class_1 = []
        
        print("Loading Image Dataset")
        for seq in tqdm(image_dataset.labels): # for each seq (num total seq,frame,chan,h,w)
            category = image_dataset.action_types[seq]
            if category in self.unique_clips:
                self.unique_clips[category] += 1
            else:
                self.unique_clips[category] = 1
            seq_data = np.stack([value for value in image_dataset.data[seq].values()], axis=0)
            if image_dataset.labels[seq] == 0:
                class_0.append((seq_data,0,category))                                                  
            elif image_dataset.labels[seq] == 1:
                class_1.append((seq_data,1,category))
        y_0 = [0]*len(class_0)  
        y_1 = [1]*len(class_1)
        min_number = min(len(class_0), len(class_1))
        return class_0, class_1, y_0, y_1, min_number

    '''Returns lists of tuples train and test each containing (data, label, category). This code assumes that all sequences are the same length.'''
    def build_real_image_dataset(self):
        image_dataset = RawImageDataset()
        image_dataset.dataset_save_path = self.config.location_data["input_path"]
        image_dataset = image_dataset.load()
        self.frame_limit = image_dataset.frame_limit
        self.color_channels = image_dataset.color_channels
        self.im_width = image_dataset.im_width
        self.im_height = image_dataset.im_height
        class_0, class_1, y_0, y_1, min_number = self.prep_dataset(image_dataset)
        
        if self.config.training_configuration['downsample']: #TODO: fix this code. this only works if class 0 is always the majority class. 
            class_0, y_0 = resample(class_0, y_0, n_samples=min_number)
        
        if self.config.location_data["transfer_path"] != None:
            test_dataset = RawImageDataset()
            test_dataset.dataset_save_path = self.config.location_data["transfer_path"]
            test_dataset = test_dataset.load()
            test_class_0, test_class_1, _, _, _ = self.prep_dataset(test_dataset)
            train_dataset = shuffle(class_0 + class_1) #training set will consist of the full training dataset
            test_dataset = shuffle(test_class_0 + test_class_1) #testing set will consist of the full transfer dataset
            return train_dataset, test_dataset
        else:
            train, test, _, _ = train_test_split(class_0+class_1, 
                                                            y_0+y_1, 
                                                            test_size=self.config.training_configuration['split_ratio'], 
                                                            shuffle=True, 
                                                            stratify=y_0+y_1, 
                                                            random_state=self.config.seed)
            return train, test

    def train(self):
        if (self.config.training_configuration['task_type'] in ['sequence_classification','collision_prediction']):
            tqdm_bar = tqdm(range(self.config.training_configuration['epochs']))
            for epoch_idx in tqdm_bar: # iterate through epoch   
                acc_loss_train = 0
                permutation = np.random.permutation(len(self.training_data)) # shuffle dataset before each epoch
                self.model.train()
                for i in range(0, len(self.training_data), self.config.training_configuration['batch_size']): # iterate through batches of the dataset
                    batch_index = i + self.config.training_configuration['batch_size'] if i + self.config.training_configuration['batch_size'] <= len(self.training_data) else len(self.training_data)
                    indices = permutation[i:batch_index]
                    batch_x = self.training_data[indices]
                    batch_x = self.toGPU(batch_x, torch.float32)
                    if self.config.training_configuration['task_type']  == 'sequence_classification': 
                      batch_y = self.training_labels[indices] #batch_x = (batch, frames, channel, h, w)
                    elif self.config.training_configuration['task_type']  == 'collision_prediction':
                      batch_y = np.concatenate([np.full(len(self.training_data[i]),self.training_labels[i]) for i in indices]) #batch_x consists of individual frames not sequences/groups of frames, batch_y extends labels of each sequence to all frames in the sequence
                    batch_y = self.toGPU(batch_y, torch.long)
                    
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
            raise ValueError('train(): task type error')
              
    def model_inference(self, X, y, clip_name):
        labels = torch.LongTensor().to(self.config.model_configuration['device'])
        outputs = torch.FloatTensor().to(self.config.model_configuration['device'])
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
                    
                    batch_x = X[i:batch_index]
                    batch_x = self.toGPU(batch_x, torch.float32)
                    if self.config.training_configuration['task_type']  == 'sequence_classification': 
                      batch_y = y[i:batch_index]  #batch_x = (batch, frames, channel, h, w)
                    elif self.config.training_configuration['task_type']  == 'collision_prediction':
                      batch_y = np.concatenate([np.full(len(X[k]),y[k]) for k in range(i,batch_index)]) #batch_x consists of individual frames not sequences/groups of frames, batch_y extends labels of each sequence to all frames in the sequence
                    batch_y = self.toGPU(batch_y, torch.long)
                    batch_clip_name = clip_name[i:batch_index]

                    output = self.model.forward(batch_x).view(-1, 2)
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
            if self.config.training_configuration['task_type']  == 'sequence_classification': 
              categories['outputs'] = torch.cat([categories['outputs'], torch.unsqueeze(outputs[i], dim=0)], dim=0)
              categories['labels'] = torch.cat([categories['labels'], torch.unsqueeze(labels[i], dim=0)], dim=0)
            elif self.config.training_configuration['task_type']  == 'collision_prediction':
              temps = [torch.unsqueeze(pred, dim=0) for pred in outputs[i*self.frame_limit: (i+1)*self.frame_limit]] #list of predictions for each frame
              for temp in temps:
                categories['outputs'] = torch.cat([categories['outputs'], temp], dim=0) #cat each prediction individually
              temps = [torch.unsqueeze(pred, dim=0) for pred in labels[i*self.frame_limit: (i+1)*self.frame_limit]] #list of labels for each frame
              for temp in temps:
                categories['labels'] = torch.cat([categories['labels'], temp], dim=0) #cat each label individually
              del temps   
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
            self.best_epoch = current_epoch if current_epoch != None else self.config.training_configuration['epochs']
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