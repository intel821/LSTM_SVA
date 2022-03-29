# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 18:20:08 2021

@author: MinYoung
"""

import torch, torchvision
import os
from tqdm import tqdm

video_time_list = [None, 
                   1733, 2839, 5828, 1522, 2344, 2153, 4557, 1519, 2702, 1749,
                   3220, 1090,  981, 1708, 2058, 2957, 1304, 1942, 2424, 1449,
                   1258, 1532, 1635, 1975, 2129, 1773, 2084, 1199, 2350, 2925,
                   3945, 2116, 1307, 1323, 2106, 2387, 1232, 3080, 1647, 2222,
                   3103, 3712, 2362, 3127, 3387, 1653, 2259, 1834, 1671, 1094,
                   2944, 1966, 3283, 3100, 1037, 1836, 2632, 5995, 1046, 2532,
                   4409, 2032, 3433, 2398, 1837, 1824, 2353, 1972, 4575, 1194,
                   2515, 3106, 1356, 1634, 1923, 2649, 2502,  739, 3414, 1724,
                   ]

class Cholec80Dataset_PNG(torch.utils.data.Dataset):
    def __init__(self, root_dir):
        
        
        for video_number, video_time in enumerate(video_time_list):
            if video_time == None:
                continue
        
            self.video_dir = f'{root_dir}/videos/video{video_number:02d}.mp4'
            self.png_dir = f'{root_dir}/pngs/video{video_number:02d}'
            os.makedirs(self.png_dir, exist_ok= True)
            
            loop = tqdm(range(video_time))
            loop.set_description(f'Converting Video Dataset to PNG Dataset')
            
            for sec in loop:
                filename = f'{root_dir}/pngs/video{video_number:02d}/{sec:04d}.png'
                torchvision.io.write_png(torchvision.transforms.functional.resize(self._load_frame(sec).squeeze().permute(2,0,1), (224, 224)), filename)
            
            
    def __len__(self):
        return self.vframes.shape[0]
    
    def __getitem__(self, index):
        return self.vframes[index], self.phases[index], self.tools[index]
    
    def _load_frame(self, time):
        frame, _, _ = torchvision.io.read_video(self.video_dir,
                                                start_pts= float(time),
                                                end_pts= float(time),
                                                pts_unit= 'sec'
                                                )
        return frame
    