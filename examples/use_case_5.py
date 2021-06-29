import numpy as np
import networkx as nx
import pandas as pd
from collections import defaultdict 
from pathlib import Path
from networkx.drawing.nx_agraph import to_agraph
import tqdm
import matplotlib
matplotlib.use("Agg")

import sys, os
sys.path.append(os.path.dirname(sys.path[0]))
import sg2vec.data.real_preprocessor as ip
from sg2vec.util.config_parser import configuration
import sg2vec.scene_graph.extraction.image_extractor as RealEx
from sg2vec.learning.util.trainer import Scenegraph_Trainer
from sg2vec.data.dataset import SceneGraphDataset
sys.modules['util'] = sg2vec.util

from sg2vec.learning.util.trainer import Scenegraph_Trainer
from sg2vec.scene_graph.relation_extractor import Relations  #use case will break when you remove relations from relation_extractor


def add_node(g, node, label):
    if node in g.nodes:
        return
    color = "white"
    # if "ego" in label.lower():
    #     color = "red"
    # elif "car" in label.lower():
    #     color = "green"
    # elif "lane" in label.lower():
    #     color = "yellow"
    g.add_node(node, label=label, style='filled', fillcolor=color)

def add_relation(g, src, relation, dst):
    g.add_edge(src, dst, object=relation, label=Relations(relation).name)

def visualize_graph(g, to_filename):
    A = to_agraph(g)
    A.layout('dot')
    A.draw(to_filename)

def parse_attn_weights(node_attns, sequences, dest_dir, visualize=False):

    original_batch = node_attns['original_batch']
    pool_perm = node_attns['pool_perm']
    pool_batch = node_attns['pool_batch']
    pool_score = node_attns['pool_score']
    batch_counter = Counter(original_batch)
    batch_deduct = {0: 0}

    colormap = matplotlib.cm.get_cmap('YlOrRd')

    node_attns_list = []
    for idx in range(1, len(sequences)):
        batch_deduct[idx] = batch_deduct[idx-1]+batch_counter[idx-1]

    node_index = {}
    node_dict= {}
    filtered_nodes = defaultdict(list)
    for idx, (p, b, s) in enumerate(zip(pool_perm, pool_batch, pool_score)):
        node_index[idx] = p - batch_deduct[b]
        inv_node_order = {v: k for k, v in sequences[b]['node_order'].items()}
        if b not in node_dict:
            node_dict[b] = []
        node_dict[b].append("%s:%f"%(inv_node_order[node_index[idx]], s))
        filtered_nodes[b].append([inv_node_order[node_index[idx]], s])
    node_attns_list.append(node_dict)

    if visualize:
        for idx in range(len(sequences)):
            scenegraph_edge_idx = sequences[idx]['edge_index'].numpy()
            scenegraph_edge_attr = sequences[idx]['edge_attr'].numpy()
            scenegraph_node_order = sequences[idx]['node_order']
            reversed_node_order = {v: k for k, v in scenegraph_node_order.items()}
            reversed_g = nx.MultiGraph()
            
            for edge_idx in range(scenegraph_edge_idx.shape[1]):
                src_idx = scenegraph_edge_idx[0][edge_idx]
                dst_idx = scenegraph_edge_idx[1][edge_idx]
                src_node_name = reversed_node_order[src_idx].name
                dst_node_name = reversed_node_order[dst_idx].name
                relation = scenegraph_edge_attr[edge_idx]

                add_node(reversed_g, src_idx, src_node_name)
                add_node(reversed_g, dst_idx, dst_node_name)
                add_relation(reversed_g, src_idx, relation, dst_idx)
            
            for node, score in filtered_nodes[idx]:
                node_idx = scenegraph_node_order[node]
                rgb_color = colormap(float(score))[:3]
                hsv_color =[str(x) for x in matplotlib.colors.rgb_to_hsv(rgb_color)]
                if node_idx not in reversed_g.nodes:
                    add_node(reversed_g, node_idx, node.name)
                reversed_g.nodes[node_idx]['fillcolor'] = ','.join(hsv_color)
                reversed_g.nodes[node_idx]['label'] += '\n' + str(round(float(score), 5))

            root_idx = None
            ego_idx = None
            for (node, data) in reversed_g.nodes(data=True):
                if root_idx and ego_idx:
                    break
                if data['label'].startswith('Ego'):
                    ego_idx = node
                    reversed_g.nodes[ego_idx]['pos'] = "0,20.0!"
                elif data['label'].startswith('Root'):
                    root_idx = node
                    reversed_g.nodes[root_idx]['pos'] = "0,-20.0!"
            # import pdb; pdb.set_trace()
            folder_name = sequences[idx]['folder_name']
            frame_num = sequences[idx]['frame_number']
            folder_path = dest_dir / folder_name
            folder_path.mkdir(exist_ok=True)
            visualize_graph(reversed_g, str(folder_path / (str(frame_num) + '.png')))
            # visualize_graph(reversed_g, "./tmp.png")
    return node_attns_list

def inspect_trainer(iterations=1):
    ''' Training the dynamic kg algorithm with different attention layer choice.'''
    
    training_config = configuration(r"C:\Users\harsi\av\sg2vec\config\learning_config.yaml",from_function = True) #replace with path to sg2vec\config\learning_config.yaml
    trainer = Scenegraph_Trainer(training_config)
    trainer.split_dataset()
    trainer.load_model()
    # outputs, labels, metric, folder_names = trainer.evaluate()
    
    
    outputs_train, labels_train, folder_names_train, acc_loss_train, attns_train, node_attns_train = trainer.inference(trainer.training_data, trainer.training_labels)
    outputs_test, labels_test, folder_names_test, acc_loss_test, attns_test, node_attns_test = trainer.inference(trainer.testing_data, trainer.testing_labels)
    

    metrics = {}
    metrics['train'] = get_metrics(outputs_train, labels_train)
    metrics['train']['loss'] = acc_loss_train

    metrics['test'] = get_metrics(outputs_test, labels_test)
    metrics['test']['loss'] = acc_loss_test

    print("\ntrain loss: " + str(acc_loss_train) + ", acc:", metrics['train']['acc'], metrics['train']['confusion'], metrics['train']['auc'], \
          "\ntest loss: " +  str(acc_loss_test) + ", acc:",  metrics['test']['acc'],  metrics['test']['confusion'], metrics['test']['auc'])


    columns = ['safe_level', 'risk_level', 'prediction', 'label', 'folder_name', 'attn_weights', 'node_attns_score']
    inspecting_result_df = pd.DataFrame(columns=columns)

    dest_dir = Path('/home/louisccc/NAS/louisccc/av/post_visualization_carla_2').resolve()
    dest_dir.mkdir(exist_ok=True)

    node_attns_train_proc = []
    for i in tqdm(range(len(trainer.training_data))):
        node_attns_train_proc += parse_attn_weights(node_attns_train[i], trainer.training_data[i]['sequence'], dest_dir, visualize=True)

    node_attns_test_proc = []
    for i in tqdm(range(len(trainer.testing_data))):
        node_attns_test_proc += parse_attn_weights(node_attns_test[i], trainer.testing_data[i]['sequence'], dest_dir, vizualize=True)

    for output, label, folder_name, attns, node_attns in zip(outputs_train, labels_train, folder_names_train, attns_train, node_attns_train_proc):
        inspecting_result_df = inspecting_result_df.append(
            {"safe_level":output[0],
             "risk_level":output[1],
             "prediction": 1 if output[1] > output[0] else 0,
             "label":label,
             "folder_name":folder_name,
             "attn_weights":{idx:value for idx, value in enumerate(attns)},
             "node_attns_score": node_attns}, ignore_index=True
        )
    
    for output, label, folder_name, attns, node_attns in zip(outputs_test, labels_test, folder_names_test, attns_test, node_attns_test_proc):
        inspecting_result_df = inspecting_result_df.append(
            {"safe_level":output[0],
             "risk_level":output[1],
             "prediction": 1 if output[1] > output[0] else 0,
             "label":label,
             "folder_name":folder_name, 
             "attn_weights":{idx:value for idx, value in enumerate(attns)},
             "node_attns_score": node_attns}, ignore_index=True
        )
    inspecting_result_df.to_csv("inspect_image.csv", index=False, columns=columns)

if __name__ == "__main__":
    inspect_trainer()
    
