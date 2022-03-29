# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 22:17:31 2021

@author: MinYoung
"""

import os

import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from dataset import Cholec80Dataset
from model import (
    ResNet50_Cholec80
    )
from loss import(
    Tool_Label_Smoothing,
    Phase_Label_Smoothing,
    )
from utils import (
    save_dict,
    plot
    )

from plot import PlotManager


# For Device
CUDA                =       torch.cuda.is_available()
DEVICE              =       torch.device('cuda' if CUDA else 'cpu')


def train(config,
          validation= False,
          ):
    
    dset_dir = f'../data/cholec80'
    
    os.makedirs(f'weights/{config.model_name}', exist_ok=True)
    
    # --------------------------------------------------- DATASET --------------------------------------------------
            
    dset = Cholec80Dataset(dset_dir, video_range= config.video_range)
    
    loader = torch.utils.data.DataLoader(dset, batch_size= config.batch_size, shuffle= True, drop_last= True)
    
    # --------------------------------------------------- MODEL -------------------------------------------------

    
    model = ResNet50_Cholec80(
        use_final_layer= False if config.train_stage == 1 else True,
        clip_length= config.clip_length).to(DEVICE)
        
    # print(model)

    if config.weights_file_name == None:
        step = 1 # For iteration saving
        plot_manager = PlotManager(config.interval)
        last_save_file_name = 'None'
        
    else:
        load_path = f'weights/{config.model_name}/{config.weights_file_name}'
        checkpoint = torch.load(load_path)
        step = checkpoint['step'] + 1
        plot_manager = checkpoint['plot_manager']
        model.load_state_dict(checkpoint['model_dict'])
        last_save_file_name = f'{checkpoint["step"]:08d}.pt'
  
    model.train()

        
    # ------------------------------------------------ OPTIMIZER -----------------------------------------------------
    
    # Optimizers & LearningRate Schedulers
    if config.train_stage == 1:
        parameters = model.stage_01.parameters()
        model.features.requires_grad_(False)
    elif config.train_stage == 2:
        parameters = model.stage_02.parameters()
        model.features.requires_grad_(False)
        model.stage_01.requires_grad_(False)
    elif config.train_stage == 3:
        parameters = model.parameters()
        
    optimizer = torch.optim.Adam(parameters, lr= config.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, gamma= config.lr_gamma, milestones= config.schedule)
    

    # ------------------------------------------------ CRITERION -----------------------------------------------------
    
    # Label Smoothing 
    label_smoothing_tool = Tool_Label_Smoothing()
    label_smoothing_phase = Phase_Label_Smoothing()
    
    # Loss for Phase & Loss for Tool
    criterion_tool = nn.BCELoss()
    criterion_phase = nn.BCELoss()

    # --------------------------------------------------- TRAIN -------------------------------------------------------    
    
    
    from math import ceil
    num_batch = ceil(len(dset) / config.batch_size)
    num_epoch = ceil((config.iteration - step) / num_batch)
    
    for epoch in range(num_epoch):
        
        loop = tqdm(loader)
        loop.set_description(f'Epoch [{epoch + 1}/{num_epoch}]')
    
        for batch_idx, (clip, tool_vector, phase) in enumerate(loop):
            
            clip = clip.to(DEVICE)
            # plot(clip) # 이미지가 잘 나오는 지 테스트하는 코드
            
            # Label Smoothing 해 주기
            tool_vector = label_smoothing_tool(tool_vector).to(DEVICE)
            phase_vector = label_smoothing_phase(phase).to(DEVICE)
                        
            # 순전파                        
            tool_vector_pred, phase_pred = model(clip)
            
            # loss 구하기
            loss_tool = criterion_tool(tool_vector_pred, tool_vector)
            loss_phase = criterion_phase(phase_pred, phase_vector)
            
            plot_manager.loss_tool += loss_tool.item()
            plot_manager.loss_phase += loss_phase.item()
            
            total_loss = loss_tool + loss_phase
            
            plot_manager.total_loss += total_loss.item()
            
            loop.set_postfix_str(f'Tool Loss = {loss_tool.item():.4f}, Phase Loss = {loss_phase.item():.4f}, '
                                 + f'Last Save File = {last_save_file_name}')
            
            model.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            plot_manager.data_dict['lr'].append(scheduler.get_last_lr()[0])
            scheduler.step()

            if step % config.interval == 0:                
                # Record Train Loss to Plot manager
                plot_manager.data_dict['total_loss'].append(plot_manager.total_loss / config.interval)
                plot_manager.data_dict['loss_tool'].append(plot_manager.loss_tool / config.interval)
                plot_manager.data_dict['loss_phase'].append(plot_manager.loss_phase /config.interval)
                plot_manager.total_loss, plot_manager.loss_tool, plot_manager.loss_phase = 0.0, 0.0, 0.0
                
                if validation:
                    _validate(dset, model, plot_manager)
                    
                
            if step % config.save_interval == 0:
                # Save File
                last_save_file_name = save_dict(f'weights/{config.model_name}/{step:08d}.pt',
                                                model,
                                                step,
                                                plot_manager,)
                
                
                plot_manager.plot_multiple_loss()
            
            if step == config.iteration:
                print('\n\n\n\n\nIteration Finished')
                break
            step += 1
            


# validation 아직 사용 안함
# Tool 같은 경우는 여러 개가 존재할 수 있어서 단순히 softmax로 처리해서
# 정확도를 측정할 수 없으므로 일단은 Phase에 대해서만 Validation을 구현함
# validation 아직 사용 안함
def _validate(config,
              dataset,
              model,
              plot_manager,
              ):

    seed = 4452
    sampler = torch.utils.data.RandomSampler(data_source= dataset,
                                             replacement= False,
                                             num_samples= len(dataset)//1000,
                                             generator= torch.Generator().manual_seed(seed))
    
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size= config.batch_size, sampler= sampler, drop_last= False)
        
    with torch.no_grad():
        
        for batch_idx, (img, tool_vector, phase) in enumerate(loader):
    
            img = img.to(DEVICE)
            phase = phase.to(DEVICE)
            tool_vector = tool_vector.to(DEVICE)
    
            tool_pred, phase_pred = model(img)
            phase_pred = F.softmax(phase_pred, dim= -1)
            
            max, indices = torch.max(phase_pred, dim= -1)
                        
            plot_manager.val_total += img.size(0)
            plot_manager.val_correct += (indices == phase).sum().cpu().numpy()
            
        plot_manager.calculate_val_acc()
        
    model.train()