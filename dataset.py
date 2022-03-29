# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 23:32:53 2021

@author: MinYoung
"""

import torch, torchvision
import os

tool_classes = ['Grasper', 'Bipolar', 'Hook', 'Scissors', 'Clipper', 'Irrigator', 'SpecimenBag']

phase_classes = {'Preparation'                : 0,
                 'CalotTriangleDissection'    : 1,
                 'ClippingCutting'            : 2,
                 'GallbladderDissection'      : 3,
                 'GallbladderPackaging'       : 4,
                 'CleaningCoagulation'        : 5,
                 'GallbladderRetraction'      : 6,               
                 }

class Cholec80Dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, clip_length= 10, video_range= [1, 80]):
                
        self.clip_length = clip_length
        self.video_range = video_range
        
        # 이미지 수만큼 담을 리스트들 만들어 놓기
        # data = (img path, tool vector, phase)
        self.data = []
        
        # 파일 주소
        self.png_dir = f'{root_dir}/pngs'
        self.tool_dir = f'{root_dir}/tool_annotations'
        self.phase_dir = f'{root_dir}/phase_annotations'
        
        # 비디오 이름 리스트로 만들기
        self.video_list = [None] + os.listdir(self.png_dir)
                
        # 이미지 파일 주소 채워넣기
        for i in range(video_range[0], video_range[1] + 1):
            
            # 비디오 파일에 대한 경로
            video_path = f'{self.png_dir}/{self.video_list[i]}'
            # 각 이미지 파일에 대한 경로
            file_names = os.listdir(video_path)
            
            # annotation 불러오기
            tools = self._get_tool_annotation(f'{self.tool_dir}/{self.video_list[i]}-tool.txt')
            phases = self._get_phase_annotation(f'{self.phase_dir}/{self.video_list[i]}-phase.txt')
                        
            # 클립 수 만큼은 맨 앞 영상은 못쓰니까 제외하기
            for idx in range(clip_length - 1, len(file_names)):
                
                # data = (tool vector, phase, video number, img number)
                self.data.append((tools[idx], phases[idx], i, idx))

                
    def __len__(self):        
        return len(self.data)
    
    def __getitem__(self, index):
        # 인덱스로 데이터를 찾아서
        tool_vector, phase, video_number, image_number = self.data[index]
        
        # 클립을 만들 이미지 리스트를 만든 후
        imgs = []
        # 클립의 길이만큼 이미지를 로드한다.
        for i in range(self.clip_length):
            # 이 때 첫 번째 이미지는 인덱스로 찾아간 이미지이다
            # 첫 번째 이미지를 기준으로 시간 상 앞에 있는 이미지를 찾아서 로드한다
            # 위에서 인덱스를 저장할 때 클립 길이보다 짧은 인덱스는 저장하지 않았다
            path = f'{self.png_dir}/{self.video_list[video_number]}/{image_number - i:04d}.png'
            imgs.append(torchvision.io.read_image(path) / 255.0)
            
        # 시간 상 뒤에서부터 저장한 꼴이니 reverse로 뒤집어 준다
        imgs.reverse()
            
        clip = torch.stack(imgs, dim= 0)
                
        return clip, tool_vector, phase
    

    def _get_tool_annotation(self, dir):
        '''
        return type : list[Video length (sec)] X tensor[Tool Num (7)]
        '''
        
        tools = []
        
        with open(dir) as f:
            lines = f.readlines()
            for i in range(1, len(lines)):
                
                tool_vector = list(map(float, lines[i].strip().split('\t')))
                tool_vector = torch.tensor(tool_vector[1:])
                tools.append(tool_vector)
                
        # print(len(tools))
                
        return tools


    def _get_phase_annotation(self, dir, fps = 25):
        '''
        return type : list[Video length (sec)]
        '''
    
        phases = []
        
        with open(dir) as f:
            lines = f.readlines()
            for i in range(1, len(lines), fps):
                
                # print(lines[i].strip().split('\t')[0])
                
                phase = torch.tensor(phase_classes[lines[i].strip().split('\t')[-1]]).to(torch.int64)
                phases.append(phase)
                
        # print(len(phases))    
        
        return phases
    