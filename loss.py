# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 14:19:10 2021

@author: MinYoung
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Phase_Label_Smoothing(nn.Module):
    
    def __init__(self, num_cls= 7,):
        super(Phase_Label_Smoothing, self).__init__()
        self.num_cls= num_cls
        
    def forward(self, targets, smoothing= 0.1):
        
        confidence = 1.0 - smoothing
        
        target_vectors = torch.zeros(targets.size(0), self.num_cls).fill_(smoothing / (self.num_cls - 1))
        target_vectors.scatter_(1, targets.data.unsqueeze(1), confidence)
        
        return target_vectors
    
    
'''
Phase 의 경우에는 classification이라 target의 확률 합이 1이 되도록 하지만
Tool 의 경우, multi-label classification 이므로 target의 합이 0이 될 수도,
1이 될 수도 2 가 될 수도 있으므로 동일한 로직으로 코드를 구현하면 Negative 예측을
뜻하는 값이 Label에 존재하는 Positive의 갯수에 따라 달라지게 될 것이므로
모든 경우에 동일하게 유지시켜 주기 위해 그냥 smoothing 값을 더하거나 빼는 것으로 코딩함
'''

class Tool_Label_Smoothing(nn.Module):
    
    def __init__(self,):
        super(Tool_Label_Smoothing, self).__init__()
        
    def forward(self, target_vectors, smoothing= 0.05):
        
        plus = target_vectors == 0.0
        minus = target_vectors == 1.0
        
        target_vectors = target_vectors + smoothing * plus - smoothing * minus
        
        return target_vectors
    
if __name__ == '__main__':
    
    pls = Phase_Label_Smoothing()
    x = torch.tensor([6, 1, 3, 0, 5, 2])
    
    out = pls(x)
    
    print(x)
    print(out)
    