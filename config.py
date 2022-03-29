# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 12:38:42 2021

@author: MinYoung
"""

class Config():
    def __init__(self, train_stage, mode):
        
        self.train_stage = train_stage
        self.mode = mode
        
        self.model_name = 'ResNet50 Cholec80'
        
        # H Params for Training
        if self.mode == 'train':
            
            self.video_range = [1, 40]
            
            if self.train_stage == 1:
                
                self.weights_file_name = None
                self.batch_size = 16
                self.lr = 1E-4
                self.lr_gamma = 0.1
                self.iteration = 1E4
                self.schedule = [5E3]
                
            elif self.train_stage == 2:
                
                self.weights_file_name = "00010000.pt"
                self.batch_size = 16
                self.lr = 1E-4
                self.lr_gamma = 0.1
                self.iteration = 2E4
                self.schedule = [15E3]
            
            elif self.train_stage == 3:
                
                self.weights_file_name = "00020000.pt"
                self.batch_size = 4
                self.lr = 1E-4
                self.lr_gamma = 0.1
                self.iteration = 1E5
                self.schedule = [6E4]
                
        elif self.mode == 'test':
            
            self.weights_file_name = "00100000.pt"
            self.batch_size = 16
            self.video_range = [41, 45]
        
        # Record interval
        self.interval = 10
        self.save_interval = 1000
        
        # For model
        self.clip_length = 10
        
        
        self.phase_label = ['Phase 1', 'Phase 2', 'Phase 3', 'Phase 4', 'Phase 5', 'Phase 6', 'Phase 7', ]
        
        
        
