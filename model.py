# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 21:55:18 2021

@author: MinYoung
"""

import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F

class ResNet50_Cholec80(nn.Module):
    def __init__(self, use_final_layer, clip_length= 10):
        super(ResNet50_Cholec80, self).__init__()
        
        self.use_final_layer = use_final_layer
        
        resnet = torchvision.models.resnet50(pretrained=True)
        self.features = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
            resnet.avgpool, nn.Flatten()
            )
        
        self.stage_01 = nn.ModuleDict({
            
            'lstm'    : nn.LSTM(2048, 512, batch_first=True),
            
            'fc_tool' : nn.Sequential(
                            nn.Linear(2048, 512),
                            nn.ReLU(inplace= True),
                            nn.Linear(512, 7),
                            nn.Sigmoid()
                            ),
            
            'fc_phase' : nn.Sequential(
                            nn.Linear(512, 7),
                            nn.Sigmoid()
                            ),
                            
            })
        
        self.stage_02 = nn.ModuleDict({
            
            'fc_final' : nn.Sequential(
                            nn.Linear(14, 14),
                            nn.Sigmoid()
                            ),
            })
        
        self._init_weights(self.stage_01)
        self._init_weights(self.stage_02)

    def forward(self, x):
        
        # N: Batch size
        # L: Sequence Length
        # C: Channel size
        # H: Height
        # W: Width
        N, L, C, H, W = x.size()
        
        input = x.view(N * L, C, H, W)
        
        out_features = self.features(input)
        out_features = out_features.view(N, L, 2048)
        
        input_tool = out_features[:,-1,:]
        out_tool = self.stage_01['fc_tool'](input_tool)
        
        self.stage_01['lstm'].flatten_parameters()
        out_lstm, _ = self.stage_01['lstm'](out_features)
        out_lstm = out_lstm[:,-1,:].contiguous()
        out_phase = self.stage_01['fc_phase'](out_lstm)        
        
        if self.use_final_layer:
            final_input = torch.cat([out_tool, out_phase], dim= -1)
            
            out = self.stage_02['fc_final'](final_input)
            out_tool, out_phase = out[:,:7], F.softmax(out[:,7:], dim= -1)
            
        return out_tool, out_phase
    
    def _init_weights(self, m):
        for module in m.modules():
            if type(module) == nn.Linear:
                nn.init.xavier_uniform_(module.weight)


if __name__ == '__main__':
    
    model = ResNet50_Cholec80(use_final_layer= True)
    x = torch.randn(4, 10, 3, 224, 224)
    
    out = model(x)
    
    print(out[0].size(), out[1].size())
    