# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 22:17:14 2021

@author: MinYoung
"""



import torch

from train import train
from test_ import (
    test,
    calculate_score,
    )
from config import Config


# Train & Test Selection
# MODE                =      'train'
MODE                =       'test'

''' 
Select Train Stage
TRAIN_STAGE
1 : Train stage 01 --> LSTM and 2 FC layers
2 : Train stage 02 --> 1 Final FC layer for Phase and Tool both
3 : Train all parameters in Network
'''

TRAIN_STAGE = 1

def main(mode):
    
    cf = Config(TRAIN_STAGE, MODE)
    
    if mode == 'train':

        train(cf)
            
    elif mode == 'test':
        result = test(cf,
                      use_pre_calculated_data= False)
        
        phase_gt, phase_preds, tool_gt, tool_preds = result
        calculate_score(phase_gt, phase_preds, tool_gt, tool_preds, cf)
          
if __name__ == '__main__':
    main(MODE)


