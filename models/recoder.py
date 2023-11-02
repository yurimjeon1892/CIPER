import torch
from torch import nn
import torch.nn.functional as F

import torchvision
import numpy as np

from .common import MLP

class Recoder(nn.Module):
    
    def __init__(self, args):
        super().__init__()
        
        # print_recoder()
        self.grd_patch_size = (int(args["grd_img_size"][0] / args["patch_size"]), int(args["grd_img_size"][1] / args["patch_size"]))
        self.arl_patch_size = (int(args["arl_img_size"][0] / args["patch_size"]), int(args["arl_img_size"][1] / args["patch_size"]))
        
        self.repeat_grd_factor = args["repeat_grd_factor"]
        self.grd_to_arl_factor = args["grd_to_arl_factor"]
        
        self.rng_grd_w = self.grd_patch_size[1] * self.repeat_grd_factor
        self.rng_arl_w = self.rng_grd_w * self.grd_to_arl_factor
        
        self.mlp_attn_grd = MLP(args["dim_embed"], int(args["dim_embed"] / 4), output_dim=args["dim_embed"], num_layers=3) 
        self.mlp_attn_arl = MLP(args["dim_embed"], int(args["dim_embed"] / 4), output_dim=args["dim_embed"], num_layers=3) 
        
    def repeat_elements(self, x, factor):
        # x = b x 1 x w x d
        b, _, w, d = x.size()
        w_new = w * factor
        x_repeated = x.unsqueeze(3).expand(b, 1, w, factor, d).reshape(b, 1, w_new, d)
        return x_repeated
    
    def arl_to_ray_feat(self, mem_arl):
        
        n = mem_arl.size(1)
        cx, cy, rad = n // 2, n // 2, n // 2 
        
        mem_arl_max = torch.empty(mem_arl.size(0), 1, self.rng_arl_w, mem_arl.size(-1))
        for i in range(self.rng_arl_w):
            theta = (i / self.rng_arl_w) * np.pi * 2
                        
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
        
        rng_grd_w_2 = self.rng_grd_w // 2
        mem_arl_f = mem_arl_max[:, :, -rng_grd_w_2:, :].flip(2)
        mem_arl_e = mem_arl_max[:, :, :rng_grd_w_2, :].flip(2)
        mem_arl_max = torch.concat([mem_arl_f, mem_arl_max, mem_arl_e], 2)
        
        return mem_arl_max
    
    def get_ray_attn(self, mem_grd_max, mem_arl_max):
        
        inputs = mem_arl_max.permute((0, 3, 1, 2)).to(mem_arl_max.device) # bs x embed_dim x 1 x w
        filters = mem_grd_max.permute((0, 3, 1, 2)).to(mem_arl_max.device) # bs x embed_dim x 1 x w'
        
        ray_attn = F.conv2d(inputs, filters, padding=0)
        ray_attn = torch.sum(ray_attn, dim=1, keepdim=True)
        ray_attn = ray_attn[:, :, :, :self.rng_arl_w]
        return ray_attn
    
    def get_bev_attn(self, ray_attn):
        # ray_attn : b x 1 x 1 x len
        bev_attn = torch.zeros((ray_attn.size(0), ray_attn.size(1), self.arl_patch_size[0], self.arl_patch_size[1]))
        
        n = self.arl_patch_size[0]
        cx, cy, rad = n // 2, n // 2, n // 2 
        
        for i in range(self.rng_arl_w):
            theta = (i / self.rng_arl_w) * np.pi * 2
            
            dx = np.cos(theta) * n / 2 
            dy = np.sin(theta) * n / 2 
            
            if np.abs(dx) >= float(1 / rad): x_values = torch.arange(cx, dx + cx, step=dx / rad)
            else: x_values = torch.full((rad, ), cx)
            if np.abs(dy) >= float(1 / rad): y_values = torch.arange(cy, dy + cy, step=dy / rad)
            else: y_values = torch.full((rad, ), cy)
            
            line_points = torch.stack((x_values[:rad], y_values[:rad]), dim=-1)
            rounded_tensor = torch.clamp(line_points.round(), 0, n - 1).long()
            
            bev_attn[:, :, rounded_tensor[:, 0], rounded_tensor[:, 1]] = ray_attn[:, :, :, i]
            
        return bev_attn
    
    def forward(self, mem_grd, mem_arl):
        """_summary_

        Args:
            mem_grd (_type_): bs x num_patches_grd x dim_embed
            mem_arl (_type_): bs x num_patches_arl x dim_embed

        Returns:
            _type_: _description_
        """
        mem_grd = self.mlp_attn_grd(mem_grd) # bs x num_patches_grd x dim_embed
        mem_arl = self.mlp_attn_arl(mem_arl) # bs x num_patches_arl x dim_embed
        
        mem_grd = mem_grd.view(mem_grd.size(0), self.grd_patch_size[0], self.grd_patch_size[1], mem_grd.size(-1))
        mem_arl = mem_arl.view(mem_arl.size(0), self.arl_patch_size[0], self.arl_patch_size[1], mem_arl.size(-1))
                
        mem_grd_max, _ = torch.max(mem_grd, dim=1, keepdim=True) # bs x 1 x w x dim_embed
        mem_grd_max = self.repeat_elements(mem_grd_max, self.repeat_grd_factor) # bs x 1 x rng_grd_w x dim_embed
        
        mem_arl_max = self.arl_to_ray_feat(mem_arl) # bs x 1 x self.rng_arl_w x dim_embed 
        
        ray_attn = self.get_ray_attn(mem_grd_max, mem_arl_max) # bs x 1 x 1 x rng_arl_w         
        bev_attn = self.get_bev_attn(ray_attn) # bs x 1 x arl_patch_size[0] x arl_patch_size[1] 
        
        bev_attn = bev_attn.flatten(2).permute((0, 2, 1)).to(mem_arl.device) # bs x num_patches_arl x 1
        # bev_attn = bev_attn + 0.5
        bev_attn_sum, _ = torch.max(bev_attn, dim=1, keepdim=True)
        bev_attn = torch.div(bev_attn, bev_attn_sum) + 0.1
                
        return bev_attn, ray_attn