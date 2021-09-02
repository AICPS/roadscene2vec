import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score, roc_auc_score, \
    roc_curve, balanced_accuracy_score, matthews_corrcoef

#this file contains functions for scoring the prediction models.

'''
#Expected Inputs:
outputs: (n, 2) FloatTensor
labels: (n,) LongTensor
'''
def get_metrics(outputs, labels):
    labels_tensor = labels.cpu()
    outputs_tensor = outputs.cpu()
    preds = outputs_tensor.max(1)[1].type_as(labels_tensor).cpu() #binarized version of outputs_tensor.

    metrics = {}
    metrics['acc'] = accuracy_score(labels_tensor, preds)
    metrics['f1'] = f1_score(labels_tensor, preds, average="binary")
    conf = confusion_matrix(labels_tensor, preds)
    metrics['fpr'] = conf[0][1] / (conf[0][1] + conf[0][0]) #FPR = FP/(FP+TN)
    metrics['tnr'] = conf[0][0] / (conf[0][1] + conf[0][0]) #TNR = TN/(FP+TN)
    metrics['fnr'] = conf[1][0] / (conf[1][0] + conf[1][1]) #FNR = FN/(FN+TP)
    metrics['confusion'] = str(conf).replace('\n', ',')
    metrics['precision'] = precision_score(labels_tensor, preds, average="binary")
    metrics['recall'] = recall_score(labels_tensor, preds, average="binary") #recall and TPR are the same. TPR = TP/(TP+FN)
    metrics['auc'] = get_auc(outputs_tensor, labels_tensor)
    metrics['label_distribution'] = str(np.unique(labels_tensor, return_counts=True)[1])
    metrics['balanced_acc'] = balanced_accuracy_score(labels_tensor, preds)
    metrics['mcc'] = matthews_corrcoef(labels_tensor, preds)
    
    return metrics 

#returns onehot version of labels. can specify n_classes to force onehot size.
def encode_onehot(labels, n_classes=None):
    if(n_classes):
        classes = set(range(n_classes))
    else:
        classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

#log data to to Weights & Biases
def log_wandb(metrics):
    wandb.log({
        "train_acc": metrics['train']['acc'],
        "val_acc": metrics['test']['acc'],
        "train_acc_balanced": metrics['train']['balanced_acc'],
        "val_acc_balanced": metrics['test']['balanced_acc'],
        "train_loss": metrics['train']['loss'],
        "val_loss": metrics['test']['loss'],
        'train_auc': metrics['train']['auc'],
        'train_f1': metrics['train']['f1'],
        'val_auc': metrics['test']['auc'],
        'val_f1': metrics['test']['f1'],
        'train_precision': metrics['train']['precision'],
        'train_recall': metrics['train']['recall'],
        'val_precision': metrics['test']['precision'],
        'val_recall': metrics['test']['recall'],
        'train_conf': metrics['train']['confusion'],
        'val_conf': metrics['test']['confusion'],
        'train_fpr': metrics['train']['fpr'],
        'train_tnr': metrics['train']['tnr'],
        'train_fnr': metrics['train']['fnr'],
        'val_fpr': metrics['test']['fpr'],
        'val_tnr': metrics['test']['tnr'],
        'val_fnr': metrics['test']['fnr'],
        'train_avg_seq_len': metrics['train']['avg_seq_len'],
        'train_avg_pred_frame': metrics['train']['avg_prediction_frame'],
        'val_avg_seq_len': metrics['test']['avg_seq_len'],
        'val_avg_pred_frame': metrics['test']['avg_prediction_frame'],
        'train_avg_pred_risky_indices': metrics['train']['avg_predicted_risky_indices'],
        'train_avg_pred_safe_indices': metrics['train']['avg_predicted_safe_indices'],
        'val_avg_pred_risky_indices': metrics['test']['avg_predicted_risky_indices'],
        'val_avg_pred_safe_indices': metrics['test']['avg_predicted_safe_indices'],
        'best_epoch': metrics['best_epoch'],
        'best_val_loss': metrics['best_val_loss'],
        'best_val_acc': metrics['best_val_acc'],
        'best_val_auc': metrics['best_val_auc'],
        'best_val_conf': metrics['best_val_conf'],
        'best_val_mcc': metrics['best_val_mcc'],
        'best_val_acc_balanced': metrics['best_val_acc_balanced'],
        'train_mcc': metrics['train']['mcc'],
        'val_mcc': metrics['test']['mcc'],
        'avg_inf_time': metrics['avg_inf_time'],
        'best_avg_pred_frame': metrics['best_avg_pred_frame'],
        # 'test_seq_tpr': metrics['test']['seq_tpr'],
        # 'test_seq_tnr': metrics['test']['seq_tnr'],
        # 'test_seq_fpr': metrics['test']['seq_fpr'],
        # 'test_seq_fnr': metrics['test']['seq_fnr'],
        # 'train_seq_tpr': metrics['train']['seq_tpr'],
        # 'train_seq_tnr': metrics['train']['seq_tnr'],
        # 'train_seq_fpr': metrics['train']['seq_fpr'],
        # 'train_seq_fnr': metrics['train']['seq_fnr']
    })
    

def log_wandb_transfer_learning(metrics):
    wandb.log({
        "val_acc": metrics['test']['acc'],
        "val_acc_balanced": metrics['test']['balanced_acc'],
        "val_loss": metrics['test']['loss'],
        'val_auc': metrics['test']['auc'],
        'val_f1': metrics['test']['f1'],
        'val_precision': metrics['test']['precision'],
        'val_recall': metrics['test']['recall'],
        'val_conf': metrics['test']['confusion'],
        'val_fpr': metrics['test']['fpr'],
        'val_tnr': metrics['test']['tnr'],
        'val_fnr': metrics['test']['fnr'],
        'val_avg_seq_len': metrics['test']['avg_seq_len'],
        'val_avg_pred_frame': metrics['test']['avg_prediction_frame'],
        'val_avg_pred_risky_indices': metrics['test']['avg_predicted_risky_indices'],
        'val_avg_pred_safe_indices': metrics['test']['avg_predicted_safe_indices'],
        'best_epoch': metrics['best_epoch'],
        'best_val_loss': metrics['best_val_loss'],
        'best_val_acc': metrics['best_val_acc'],
        'best_val_auc': metrics['best_val_auc'],
        'best_val_conf': metrics['best_val_conf'],
        'best_val_mcc': metrics['best_val_mcc'],
        'best_val_acc_balanced': metrics['best_val_acc_balanced'],
        'val_mcc': metrics['test']['mcc'],
        'avg_inf_time': metrics['avg_inf_time'],
        'best_avg_pred_frame': metrics['best_avg_pred_frame'],
        'test_seq_tpr': metrics['test']['seq_tpr'],
        'test_seq_tnr': metrics['test']['seq_tnr'],
        'test_seq_fpr': metrics['test']['seq_fpr'],
        'test_seq_fnr': metrics['test']['seq_fnr']
    })

def log_im_wandb(metrics):
    wandb.log({
        "train_acc": metrics['train']['acc'],
        "val_acc": metrics['test']['acc'],
        "train_acc_balanced": metrics['train']['balanced_acc'],
        "val_acc_balanced": metrics['test']['balanced_acc'],
        "train_loss": metrics['train']['loss'],
        "val_loss": metrics['test']['loss'],
        'train_auc': metrics['train']['auc'],
        'train_f1': metrics['train']['f1'],
        'val_auc': metrics['test']['auc'],
        'val_f1': metrics['test']['f1'],
        'train_precision': metrics['train']['precision'],
        'train_recall': metrics['train']['recall'],
        'val_precision': metrics['test']['precision'],
        'val_recall': metrics['test']['recall'],
        'train_conf': metrics['train']['confusion'],
        'val_conf': metrics['test']['confusion'],
        'train_fpr': metrics['train']['fpr'],
        'train_tnr': metrics['train']['tnr'],
        'train_fnr': metrics['train']['fnr'],
        'val_fpr': metrics['test']['fpr'],
        'val_tnr': metrics['test']['tnr'],
        'val_fnr': metrics['test']['fnr'],
        'train_avg_seq_len': metrics['train']['avg_seq_len'],
        'val_avg_seq_len': metrics['test']['avg_seq_len'],
        'best_epoch': metrics['best_epoch'],
        'best_val_loss': metrics['best_val_loss'],
        'best_val_acc': metrics['best_val_acc'],
        'best_val_auc': metrics['best_val_auc'],
        'best_val_conf': metrics['best_val_conf'],
        'best_val_mcc': metrics['best_val_mcc'],
        'best_val_acc_balanced': metrics['best_val_acc_balanced'],
        'train_mcc': metrics['train']['mcc'],
        'val_mcc': metrics['test']['mcc'],
        'avg_inf_time': metrics['avg_inf_time'],
    })

def log_wandb_categories(metrics, id):
    wandb.log({
        "train_acc"+"_"+id: metrics['train'][id]['acc'],
        "val_acc"+"_"+id: metrics['test'][id]['acc'],
        "train_acc_balanced"+"_"+id: metrics['train'][id]['balanced_acc'],
        "val_acc_balanced"+"_"+id: metrics['test'][id]['balanced_acc'],
        'train_auc'+"_"+id: metrics['train'][id]['auc'],
        'train_f1'+"_"+id: metrics['train'][id]['f1'],
        'val_auc'+"_"+id: metrics['test'][id]['auc'],
        'val_f1'+"_"+id: metrics['test'][id]['f1'],
        'train_precision'+"_"+id: metrics['train'][id]['precision'],
        'train_recall'+"_"+id: metrics['train'][id]['recall'],
        'val_precision'+"_"+id: metrics['test'][id]['precision'],
        'val_recall'+"_"+id: metrics['test'][id]['recall'],
        'train_conf'+"_"+id: metrics['train'][id]['confusion'],
        'val_conf'+"_"+id: metrics['test'][id]['confusion'],
        'train_fpr'+"_"+id: metrics['train'][id]['fpr'],
        'train_tnr'+"_"+id: metrics['train'][id]['tnr'],
        'train_fnr'+"_"+id: metrics['train'][id]['fnr'],
        'val_fpr'+"_"+id: metrics['test'][id]['fpr'],
        'val_tnr'+"_"+id: metrics['test'][id]['tnr'],
        'val_fnr'+"_"+id: metrics['test'][id]['fnr'],
        'train_mcc'+"_"+id: metrics['train'][id]['mcc'],
        'val_mcc'+"_"+id: metrics['test'][id]['mcc'],
})
#~~~~~~~~~~Scoring Metrics~~~~~~~~~~
#note: these scoring metrics only work properly for binary classification use cases (graph classification, dyngraph classification) 
def get_auc(outputs, labels):
    try:    
        labels = encode_onehot(labels.numpy().tolist(), 2) #binary labels
        auc = roc_auc_score(labels, outputs.numpy(), average="micro")
    except ValueError as err: 
        print("error calculating AUC: ", err)
        auc = 0.0
    return auc

#NOTE: ROC curve is only generated for positive class (risky label) confidence values 
#render parameter determines if the figure is actually generated. If false, it saves the values to a csv file.
def get_roc_curve(outputs, labels, render=False):
    risk_scores = []
    outputs = preprocessing.normalize(outputs.numpy(), axis=0)
    for i in outputs:
        risk_scores.append(i[1])
    fpr, tpr, thresholds = roc_curve(labels.numpy(), risk_scores)
    roc = pd.DataFrame()
    roc['fpr'] = fpr
    roc['tpr'] = tpr
    roc['thresholds'] = thresholds
    roc.to_csv("ROC_data.csv")

    if(render):
        plt.figure(figsize=(8,8))
        plt.xlim((0,1))
        plt.ylim((0,1))
        plt.ylabel("TPR")
        plt.xlabel("FPR")
        plt.title("Receiver Operating Characteristic")
        plt.plot([0,1],[0,1], linestyle='dashed')
        plt.plot(fpr,tpr, linewidth=2)
        plt.savefig("ROC_curve.svg")