import torch
from torch import nn

from functools import partial

import torchvision
import numpy as np

from .common import MLP
from common.utils import print_recoder

class Recoder(nn.Module):
    
    def __init__(self, args, range_width):
        super().__init__()
        
        # print_recoder()
        self.grd_patch_size = (int(args["grd_img_size"][0] / args["patch_size"]), int(args["grd_img_size"][1] / args["patch_size"]))
        self.arl_patch_size = (int(args["arl_img_size"][0] / args["patch_size"]), int(args["arl_img_size"][1] / args["patch_size"]))
        
        self.range_width = range_width
    
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
        
        range_score = torch.zeros(mem_arl.size(0), self.range_width, mem_arl.size(-1))
        return range_score