# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 22:18:18 2021

@author: MinYoung
"""


import matplotlib.pyplot as plt

class PlotManager():
    def __init__(self, interval,
                 record_opt_dict= {
            
                    'lr' : True,
                    'total_loss' : True,
                    'loss_tool' : True,
                    'loss_phase' : True,
                    
                    }):
        
        self.record_opt_dict = record_opt_dict
        self.interval = interval
        self.data_dict = {
            
            'lr' : [],
            'total_loss' : [],
            'loss_tool' : [],
            'loss_phase' : [],
            
            }
        
        self.total_loss = 0.0
        self.loss_tool = 0.0
        self.loss_phase = 0.0
        
    def plot(self, name):
        
        if self.record_opt_dict[name]:
                        
            x = range(1, (len(self.data_dict[name]) + 1))
            plt.plot(x, self.data_dict[name],)
            
            if name == 'lr':
                plt.xlabel('Iteration')
            else:
                plt.xlabel(f'Iteration (x{self.interval})')
                
            plt.ylabel('Value')
            plt.title(name.upper())
            plt.legend()
            plt.show()
            plt.close()
        
        else:
            print(f'Not including {name} information.')
        
    def plot_all(self):    
        for key in self.record_opt_dict.keys():
            if self.record_opt_dict[key]:
                self.plot(key)
    
    def plot_multiple_loss(self):
        
        x = range(1, (len(self.data_dict['total_loss']) + 1))
        plt.plot(x, self.data_dict['loss_tool'], label= 'Loss_Tool')
        plt.plot(x, self.data_dict['loss_phase'], label= 'Loss_Phase')
        
        plt.xlabel(f'Iteration (x{self.interval})')
        plt.ylabel('Value')
        plt.title('Loss')
        plt.legend()
        plt.show()
        plt.close()