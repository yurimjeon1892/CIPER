import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

from .common import MLP

class Recoder(nn.Module):
    
    def __init__(self, args):
        super().__init__()
        self.grd_patch_size = (int(args["grd_img_size"][0] / args["patch_size"]), int(args["grd_img_size"][1] / args["patch_size"]))
        self.arl_patch_size = (int(args["arl_img_size"][0] / args["patch_size"]), int(args["arl_img_size"][1] / args["patch_size"]))
        
        self.repeat_grd_factor = args["repeat_grd_factor"]
        self.grd_to_arl_factor = args["grd_to_arl_factor"]
        
        self.grd_feat_width = self.grd_patch_size[1] * self.repeat_grd_factor
        self.arl_feat_width = self.grd_feat_width * self.grd_to_arl_factor
        
        self.mlp_mask_grd = MLP(args["dim_embed"], int(args["dim_embed"] / 4), output_dim=args["dim_embed"], num_layers=3) 
        self.mlp_mask_arl = MLP(args["dim_embed"], int(args["dim_embed"] / 4), output_dim=args["dim_embed"], num_layers=3) 
        
        self.center_offset = 2 # fixed
        
    def repeat_elements(self, x, factor):
        b, _, w, d = x.size()
        w_new = w * factor
        x_repeated = x.unsqueeze(3).expand(b, 1, w, factor, d).reshape(b, 1, w_new, d)
        return x_repeated
    
    def convert_arl_to_rng_feat(self, mem_arl):        
        n = mem_arl.size(1)
        cx, cy, rad = (n // 2), (n // 2), n 
        
        rng_feat = torch.empty(mem_arl.size(0), 1, self.arl_feat_width, mem_arl.size(-1))
        for i in range(self.arl_feat_width):
            theta = (i / self.arl_feat_width) * np.pi * 2
                        
            dx = np.cos(theta) * rad
            dy = np.sin(theta) * rad
            
            if self.center_offset < dx: x_values = torch.arange(cx + self.center_offset - 1, cx + dx, step=(dx - self.center_offset + 1) / rad)            
            elif 0 < dx and dx <= self.center_offset : x_values = torch.full((rad, ), cx)
            elif -self.center_offset < dx and dx <= 0: x_values = torch.full((rad, ), cx - 1)
            elif dx <= -self.center_offset: x_values = torch.arange(cx - self.center_offset, cx + dx, step=(dx + self.center_offset) / rad)
            
            if self.center_offset < dy: y_values = torch.arange(cy + self.center_offset - 1, cy + dy, step=(dy - self.center_offset + 1) / rad)            
            elif 0 < dy and dy <= self.center_offset : y_values = torch.full((rad, ), cy)
            elif -self.center_offset < dy and dy <= 0: y_values = torch.full((rad, ), cy - 1)
            elif dy <= -self.center_offset: y_values = torch.arange(cy - self.center_offset, cy + dy, step=(dy + self.center_offset) / rad)   
            
            line_points = torch.stack((x_values[:rad], y_values[:rad]), dim=-1)
            rounded_tensor = torch.clamp(line_points.round(), 0, n - 1).long()
            
            extracted_values = mem_arl[:, rounded_tensor[:, 0], rounded_tensor[:, 1], :]
            ext_max, _ = torch.max(extracted_values, dim=1, keepdim=True)
            
            rng_feat[:, :, i, :] = ext_max.to(rng_feat.device)
        
        grd_feat_width_2 = self.grd_feat_width // 2
        rng_feat_f = rng_feat[:, :, -grd_feat_width_2:, :].flip(2)
        rng_feat_e = rng_feat[:, :, :grd_feat_width_2, :].flip(2)
        rng_feat = torch.concat([rng_feat_f, rng_feat, rng_feat_e], 2)
        return rng_feat
    
    def get_rng_mask(self, mem_grd_max, mem_arl_max):
        inputs = mem_arl_max.permute((0, 3, 1, 2)).to(mem_arl_max.device) # bs x embed_dim x 1 x w
        filters = mem_grd_max.permute((0, 3, 1, 2)).unsqueeze(1).to(mem_arl_max.device) # bs x 1 x embed_dim x 1 x w'
        
        rng_masks = []
        for b in range(inputs.size(0)):
            rng_mask = F.conv2d(inputs[b], filters[b], stride=1, padding=0)
            rng_mask = rng_mask / filters.size(-1)
            rng_masks.append(rng_mask)        
        rng_masks = torch.cat(rng_masks, 0)
        rng_masks = rng_masks[:, :, :self.arl_feat_width]
        rng_masks = torch.sigmoid(rng_masks)
        
        return rng_masks
    
    def convert_rng_to_bev_mask(self, rng_mask):       
        
        rng_mask = rng_mask.clone()
        rng_mask = (rng_mask - torch.min(rng_mask)) / (torch.max(rng_mask) - torch.min(rng_mask))
               
        bev_mask = torch.zeros((rng_mask.size(0), rng_mask.size(1), self.arl_patch_size[0], self.arl_patch_size[1]))
        
        n = self.arl_patch_size[0]
        cx, cy, rad = (n // 2), (n // 2), n 
        
        for i in range(self.arl_feat_width):
            theta = (i / self.arl_feat_width) * np.pi * 2
            
            dx = np.cos(theta) * rad
            dy = np.sin(theta) * rad
            
            if self.center_offset < dx: x_values = torch.arange(cx + self.center_offset - 1, cx + dx, step=(dx - self.center_offset + 1) / rad)            
            elif 0 < dx and dx <= self.center_offset : x_values = torch.full((rad, ), cx)
            elif -self.center_offset < dx and dx <= 0: x_values = torch.full((rad, ), cx - 1)
            elif dx <= -self.center_offset: x_values = torch.arange(cx - self.center_offset, cx + dx, step=(dx + self.center_offset) / rad)
            
            if self.center_offset < dy: y_values = torch.arange(cy + self.center_offset - 1, cy + dy, step=(dy - self.center_offset + 1) / rad)            
            elif 0 < dy and dy <= self.center_offset : y_values = torch.full((rad, ), cy)
            elif -self.center_offset < dy and dy <= 0: y_values = torch.full((rad, ), cy - 1)
            elif dy <= -self.center_offset: y_values = torch.arange(cy - self.center_offset, cy + dy, step=(dy + self.center_offset) / rad)            
            
            line_points = torch.stack((x_values[:rad], y_values[:rad]), dim=-1)
            rounded_tensor = torch.clamp(line_points.round(), 0, n - 1).long()
            
            bev_mask[:, :, rounded_tensor[:, 0], rounded_tensor[:, 1]] = rng_mask[:, :, i].unsqueeze(-1)
        
        bev_mask[:, :, cx - self.center_offset : cx + self.center_offset, cy - self.center_offset: cy + self.center_offset] = 1
        return bev_mask
    
    def forward(self, mem_grd, mem_arl):
        """_summary_

        Args:
            mem_grd (_type_): bs x num_patches_grd x dim_embed
            mem_arl (_type_): bs x num_patches_arl x dim_embed

        Returns:
            _type_: _description_
        """
        mem_grd = self.mlp_mask_grd(mem_grd) # bs x num_patches_grd x dim_embed
        mem_arl = self.mlp_mask_arl(mem_arl) # bs x num_patches_arl x dim_embed
        
        mem_grd = mem_grd.view(mem_grd.size(0), self.grd_patch_size[0], self.grd_patch_size[1], mem_grd.size(-1))
        mem_arl = mem_arl.view(mem_arl.size(0), self.arl_patch_size[0], self.arl_patch_size[1], mem_arl.size(-1))
                
        mem_grd_max, _ = torch.max(mem_grd, dim=1, keepdim=True) # bs x 1 x w x dim_embed
        mem_grd_max = self.repeat_elements(mem_grd_max, self.repeat_grd_factor) # bs x 1 x grd_feat_width x dim_embed
        
        mem_arl_max = self.convert_arl_to_rng_feat(mem_arl) # bs x 1 x arl_feat_width x dim_embed 
        
        rng_mask = self.get_rng_mask(mem_grd_max, mem_arl_max) # bs x 1 x arl_feat_width       
        bev_mask = self.convert_rng_to_bev_mask(rng_mask) # bs x 1 x arl_patch_size[0] x arl_patch_size[1] 
        
        bev_mask = bev_mask.flatten(2).permute((0, 2, 1)).to(mem_arl.device) # bs x num_patches_arl x 1
                
        return bev_mask, rng_mask