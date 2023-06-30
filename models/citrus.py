# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn.functional as F
from torch import nn

from .encdec import build_encoder, build_decoder
from .soft_triplet import SoftTripletBiLoss

class CITRUS(nn.Module):
    """ This is the CITRUS module that performs cross-view image geo-localization """
    def __init__(self, args):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        
        self.query_net = build_encoder(args)
        self.reference_net = build_encoder(args)   
        
        self.is_loc = args["task"] == "IL"
        if self.is_loc: self.pose_net = build_decoder(args)

    def forward(self, im_grnd, im_arl):
        
        out_emb_grnd, memory_grnd = self.query_net(im_grnd)
        out_emb_arl, memory_arl = self.reference_net(im_arl)      
        outputs = {
            "grnd": out_emb_grnd,
            "arl": out_emb_arl,
        }
          
        if self.is_loc: 
            out_pos = self.pose_net(memory_grnd, memory_arl)
            outputs.update(out_pos)
        
        return outputs

class SetCriterion(nn.Module):
    def __init__(self, args):
        """ Create the criterion.
        Parameters:
        """
        super().__init__()
        
        self.is_loc = args["task"] == "IL"
                
        self.losses = args["losses"]
        self.weight_dict = args["lambda"]
        
        self.device = args["device"]
        
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
        loss_dict["retrieval"] = loss_ir * self.weight_dict["retrieval"]
            
        if self.is_loc :            
            bs, n = outputs['pred_logits'].shape[:2]
            
            src_logits = outputs['pred_logits']
            target_classes = torch.cat([v["labels"] for v in targets]).view((bs, n, -1)).float().to(self.device)
            loss_ce = F.binary_cross_entropy(src_logits, target_classes) # 이거 소프트맥스인데 시그무이드로 바꿔야되는지확인 그리고 이니셜라이즈도확인!!
            loss_dict["class"] = loss_ce * self.weight_dict["class"]      
                
            src_boxes = outputs['pred_boxes']            
            target_boxes = torch.cat([v["boxes"] for v in targets]).view((bs, n, -1)).to(self.device)     
            loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
            
            p_mask = target_classes[:, :, 0] > 0
            loss_bbox = loss_bbox[p_mask].sum()
            loss_dict["bbox"] = loss_bbox * self.weight_dict["bbox"]   

        return loss_dict    
        
def build_model(args):    
    return CITRUS(args).to(args["device"])

def build_criterion(args):  
    return SetCriterion(args).to(args["device"])