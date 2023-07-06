# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn.functional as F
import math
from scipy.stats import truncnorm
from torch import nn

from .encdec import build_encoder, build_decoder
from .soft_triplet import SoftTripletBiLoss

class CITRUS(nn.Module):
    """ This is the CITRUS module that performs cross-view image geo-localization """
    def __init__(self, args, is_loc_task):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
        """
        super().__init__()
        
        self.query_net = build_encoder(args)
        self.reference_net = build_encoder(args)   
        
        self.is_loc = is_loc_task
        if self.is_loc: self.pose_net = build_decoder(args)
        
        self._initialize_weights()
                        
    def _initialize_weights(self):        
        for m in self.modules():            
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, im_grnd, im_arl):
        
        embed_grnd, memory_grnd, _, pos_grnd = self.query_net(im_grnd)
        embed_arl, memory_arl, mask, pos_arl = self.reference_net(im_arl)      
        outputs = {
            "grnd": embed_grnd,
            "arl": embed_arl,
        }
          
        if self.is_loc: 
            out_pos = self.pose_net((memory_grnd, pos_grnd), (memory_arl, pos_arl), mask)
            outputs.update(out_pos)
        
        return outputs

class SetCriterion(nn.Module):
    def __init__(self, args, is_loc_task):
        """ Create the criterion.
        Parameters:
        """
        super().__init__()
        
        self.losses = args["losses"]
        self.weight_dict = args["lambda"]
        
        self.is_loc = is_loc_task
        
        self.loss_retrieval = SoftTripletBiLoss()

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """        

        # Compute all the requested losses
        loss_dict = {}        
        loss_ir, mean_p, mean_n = self.loss_retrieval(outputs["grnd"], outputs["arl"])
        loss_dict["ir"] = loss_ir * self.weight_dict["ir"]
                    
        if self.is_loc :            
            bs, n = outputs['pred_logits'].shape[:2]
            
            src_logits = outputs['pred_logits']
            target_classes = torch.cat([v["labels"] for v in targets]).view((bs, n, -1)).float().to(src_logits.device)
            loss_ce = F.binary_cross_entropy_with_logits(src_logits, target_classes) 
            loss_dict["loc/class"] = loss_ce * self.weight_dict["loc/class"]    
                
            src_boxes = outputs['pred_boxes']            
            target_boxes = torch.cat([v["boxes"] for v in targets]).view((bs, n, -1)).to(src_boxes.device)     
            loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
            
            p_mask = target_classes[:, :, 0] > 0
            loss_bbox = loss_bbox[p_mask].sum()
            loss_dict["loc/bbox"] = loss_bbox * self.weight_dict["loc/bbox"]   

        return loss_dict    
        
def build_model(args, is_loc_task, device):    
    return CITRUS(args, is_loc_task).to(device)

def build_criterion(args, is_loc_task, device):  
    return SetCriterion(args, is_loc_task).to(device)