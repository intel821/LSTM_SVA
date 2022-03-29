# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 22:44:35 2021

@author: MinYoung
"""

import os

import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from dataset import Cholec80Dataset
from model import (
    ResNet50_Cholec80,
    )
from utils import (
    save_dict,
    plot
    )

from plot import PlotManager


# For Device
CUDA                =       torch.cuda.is_available()
DEVICE              =       torch.device('cuda' if CUDA else 'cpu')


def test(config,
         use_pre_calculated_data,
         ):
    
    if use_pre_calculated_data:
        result = torch.load('test_result.pt')
        return result['phase_gt'], result['phase_preds'], result['tool_gt'], result['tool_preds']
        
    else:
        dset_dir = f'../data/cholec80'
        
        # --------------------------------------------------- DATASET --------------------------------------------------
                
        dset = Cholec80Dataset(dset_dir, video_range= config.video_range)
        loader = torch.utils.data.DataLoader(dset, batch_size= config.batch_size, shuffle= False, drop_last= False)
        
        # --------------------------------------------------- MODEL -------------------------------------------------
        
    
        model = ResNet50_Cholec80(
            use_final_layer= False if config.train_stage == 1 else True,
            clip_length= config.clip_length).to(DEVICE)
            
        # print(model)
    
        load_path = f'weights/{config.model_name}/{config.weights_file_name}'
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint['model_dict'])
      
        model.eval()
    
        # --------------------------------------------------- TEST -------------------------------------------------------    
        
        
        from math import ceil
        num_batch = ceil(len(dset) / config.batch_size)
    
        loop = tqdm(loader)
        loop.set_description(f'Precision & Recall Testing')
    
        phase_gt = []
        phase_preds = []
        tool_gt = []
        tool_preds = []
        
    
        with torch.no_grad():
            for batch_idx, (clip, tool_vector, phase) in enumerate(loop):
                
                phase_gt.append(phase)
                tool_gt.append(tool_vector)
                
                clip = clip.to(DEVICE)        
                phase = phase.to(DEVICE)
                tool_vector = tool_vector.to(DEVICE)

                tool_vector_pred, phase_pred = model(clip)
                    
                _, phase_indices = torch.max(phase_pred, dim= -1)
                tool_pred = (tool_vector_pred > 0.5)
                
                phase_preds.append(phase_indices.detach().cpu())
                tool_preds.append(tool_pred.detach().cpu())
                
                
            phase_gt = torch.cat(phase_gt, dim= 0)
            phase_preds = torch.cat(phase_preds, dim= 0)
            tool_gt = torch.cat(tool_gt, dim= 0)
            tool_preds = torch.cat(tool_preds, dim= 0)
            
            
        torch.save({
            'phase_gt' : phase_gt,
            'phase_preds' : phase_preds,
            'tool_gt' : tool_gt,
            'tool_preds' : tool_preds,
            }, 'test_result.pt')
            
        return phase_gt, phase_preds, tool_gt, tool_preds
        
        
def calculate_score(phase_gt, phase_preds, tool_gt, tool_preds, config, weighted= False):
    
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    cf_matrix = confusion_matrix(phase_gt, phase_preds)
    print(cf_matrix)
    
    ax = sns.heatmap(cf_matrix, annot=True, fmt= "d", xticklabels= config.phase_label, yticklabels= config.phase_label,
                     cmap= sns.light_palette("seagreen", as_cmap=True))
    ax.set_title('Confusion Matrix of Phase Recognition')
    ax.set_xlabel('Ground Truth', loc= 'right')
    ax.set_ylabel('Prediction', loc= 'bottom')
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    
    from sklearn.metrics import precision_score, recall_score, accuracy_score, average_precision_score, f1_score
    
    if weighted:
        a = accuracy_score(phase_gt, phase_preds,)
        b = precision_score(phase_gt, phase_preds, average= 'weighted')
        c = recall_score(phase_gt, phase_preds, average= 'weighted')
        d = average_precision_score(tool_gt, tool_preds, average= 'weighted')
        
        print(f'Total Phase Accuracy \t\t\t\t\t: \t {100 * a:.2f} %')
        print(f'Weighted Average of Phase Precision \t: \t {100 * b:.2f} %')
        print(f'Weighted Average of Phase Recall \t\t: \t {100 * c:.2f} %')
        print(f'Weighted mAP of Tool Detection \t\t\t: \t {100 * d:.2f} %')
    
    else:
        a = accuracy_score(phase_gt, phase_preds,)
        b = precision_score(phase_gt, phase_preds, average= None)
        c = recall_score(phase_gt, phase_preds, average= None)
        d = average_precision_score(tool_gt, tool_preds, average= None)
        
        
        b, b_ = torch.std_mean(torch.tensor(b), dim= 0)
        c, c_ = torch.std_mean(torch.tensor(c), dim= 0)
        d, d_ = torch.std_mean(torch.tensor(d), dim= 0)
        
        print(f'Total Phase Accuracy \t: \t {100 * a:.2f} %')
        print(f'Phase Precision \t\t: \t {100 * b_:.2f} +- {100 * b:.2f} %')
        print(f'Phase Recall \t\t\t: \t {100 * c_:.2f} +- {100 * c:.2f} %')
        print(f'mAP of Tool Detection \t: \t {100 * d_:.2f} +- {100 * d:.2f} %')