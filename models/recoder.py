import torch
from torch import nn
import torch.nn.functional as F

import torchvision
import numpy as np

from common.utils import print_recoder

class Recoder(nn.Module):
    
    def __init__(self, args):
        super().__init__()
        
        # print_recoder()
        self.grd_patch_size = (int(args["grd_img_size"][0] / args["patch_size"]), int(args["grd_img_size"][1] / args["patch_size"]))
        self.arl_patch_size = (int(args["arl_img_size"][0] / args["patch_size"]), int(args["arl_img_size"][1] / args["patch_size"]))
        
        self.repeat_grd_factor = args["repeat_grd_factor"]
        self.grd_to_arl_factor = args["grd_to_arl_factor"]
        
    def repeat_elements(self, x, factor):
        # x = b x 1 x w x d
        b, _, w, d = x.size()
        w_new = w * factor
        x_repeated = x.unsqueeze(3).expand(b, 1, w, factor, d).reshape(b, 1, w_new, d)
        return x_repeated
    
    def extract_ray_feature(self, mem_arl, rng_grd_w):
        
        n = mem_arl.size(1)
        cx, cy, rad = n // 2, n // 2, n // 2 
        
        rng_arl_w = rng_grd_w * self.grd_to_arl_factor
        rng_grd_w_2 = rng_grd_w // 2
        
        mem_arl_max = torch.empty(mem_arl.size(0), 1, rng_arl_w, mem_arl.size(-1))
        for i in range(rng_arl_w):
            theta = (np.pi * 2 / rng_arl_w) * float(i)
                        
            dx = np.cos(theta) * n / 2 
            dy = np.sin(theta) * n / 2 
            
            if np.abs(dx) >= float(1 / rad): x_values = torch.arange(cx, dx + cx, step=dx / rad)
            else: x_values = torch.full((rad, ), cx)
            if np.abs(dy) >= float(1 / rad): y_values = torch.arange(cy, dy + cy, step=dy / rad)
            else: y_values = torch.full((rad, ), cy)
            
            line_points = torch.stack((x_values[:rad], y_values[:rad]), dim=-1)
            rounded_tensor = torch.clamp(line_points.round(), 0, n - 1).long()
            
            extracted_values = mem_arl[:, rounded_tensor[:, 0], rounded_tensor[:, 1], :]
            ext_max, _ = torch.max(extracted_values, dim=1, keepdim=True)
            
            mem_arl_max[:, :, i, :] = ext_max.to(mem_arl_max.device)
        
        mem_arl_f = mem_arl_max[:, :, -rng_grd_w_2:, :].flip(2)
        mem_arl_e = mem_arl_max[:, :, :rng_grd_w_2, :].flip(2)
        mem_arl_max = torch.concat([mem_arl_f, mem_arl_max, mem_arl_e], 2)
        
        return mem_arl_max
    
    def compute_ray_attention(self, mem_grd_max, mem_arl_max):
        ray_attn = []
        for b in range(mem_arl_max.size(0)):
            inputs = mem_arl_max[b].unsqueeze(0).permute((0, 3, 1, 2)).to(mem_arl_max.device) # 1 x embed_dim x 1 x w
            filters = mem_grd_max[b].unsqueeze(0).permute((0, 3, 1, 2)).to(mem_arl_max.device) # 1 x embed_dim x 1 x w'
            score_b = F.conv2d(inputs, filters, padding=0)
            ray_attn.append(score_b)
        ray_attn = torch.concat(ray_attn, 0)
        return ray_attn
    
    def generate_ray_attention(self, ray_attn):
        # ray_attn : b x 1 x 1 x len
        bev_ray_attn = torch.zeros((ray_attn.size(0), ray_attn.size(1), self.arl_patch_size[0], self.arl_patch_size[1]))
        
        n = self.arl_patch_size[0]
        cx, cy, rad = n // 2, n // 2, n // 2 
        
        for i in range(ray_attn.size(-1)):
            theta = (np.pi * 2 / ray_attn.size(-1)) * float(i)
            
            dx = np.cos(theta) * n / 2 
            dy = np.sin(theta) * n / 2 
            
            if np.abs(dx) >= float(1 / rad): x_values = torch.arange(cx, dx + cx, step=dx / rad)
            else: x_values = torch.full((rad, ), cx)
            if np.abs(dy) >= float(1 / rad): y_values = torch.arange(cy, dy + cy, step=dy / rad)
            else: y_values = torch.full((rad, ), cy)
            
            line_points = torch.stack((x_values[:rad], y_values[:rad]), dim=-1)
            rounded_tensor = torch.clamp(line_points.round(), 0, n - 1).long()
            
            bev_ray_attn[:, :, rounded_tensor[:, 0], rounded_tensor[:, 1]] = ray_attn[:, :, :, i]
            
        return bev_ray_attn
    
    def forward(self, mem_grd, mem_arl):
        """_summary_

        Args:
            mem_grd (_type_): bs x num_patches_grd x dim_embed
            mem_arl (_type_): bs x num_patches_arl x dim_embed

        Returns:
            _type_: _description_
        """
        mem_grd = mem_grd.view(mem_grd.size(0), self.grd_patch_size[0], self.grd_patch_size[1], mem_grd.size(-1))
        mem_arl = mem_arl.view(mem_arl.size(0), self.arl_patch_size[0], self.arl_patch_size[1], mem_arl.size(-1))
                
        mem_grd_max, _ = torch.max(mem_grd, dim=1, keepdim=True) # bs x 1 x w x dim_embed
        mem_grd_max = self.repeat_elements(mem_grd_max, self.repeat_grd_factor) # bs x 1 x w' x dim_embed
        # print(mem_grd_max.size()); exit()
        
        rng_grd_w = mem_grd_max.size(2)
        mem_arl_max = self.extract_ray_feature(mem_arl, rng_grd_w)
        # print("mem_arl_max: ", mem_arl_max.size())        
        
        ray_attn = self.compute_ray_attention(mem_grd_max, mem_arl_max)
        # print("ray_attn: ", ray_attn.size())
        
        bev_ray_attn = self.generate_ray_attention(ray_attn)
        # print("bev_ray_attn: ", bev_ray_attn.size())
        
        bev_ray_attn = bev_ray_attn.flatten(2).permute((0, 2, 1)).to(mem_arl.device)
        # print("bev_ray_attn: ", bev_ray_attn.size())
        bev_ray_attn = F.normalize(bev_ray_attn)
        
        return bev_ray_attn