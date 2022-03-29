# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 13:03:23 2021

@author: MinYoung
"""

import torch
from matplotlib import pyplot as plt

def save_dict(directory, model, step, plot_manager):
    torch.save({
                'model_dict' : model.state_dict(),
                'step' : step,
                'plot_manager' : plot_manager,
                
                }, directory)
    
    return f'{step:08d}.pt'



def plot(tensor_3d):
    
    for i in range(tensor_3d.shape[1]):
        plt.imshow(tensor_3d[0][i].detach().cpu().permute(1,2,0))
        plt.show()


    