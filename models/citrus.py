# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn.functional as F
import math
from scipy.stats import truncnorm
from torch import nn

from .encdec import build_encoder, build_decoder
from .matcher import build_matcher
from .soft_triplet import SoftTripletBiLoss

class CITRUS(nn.Module):
    """ This is the CITRUS module that performs cross-view image geo-localization """
    def __init__(self, args, is_local):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
        """
        super().__init__()
        
        self.query_net = build_encoder(args)
        self.reference_net = build_encoder(args)   
        
        self.is_loc = is_local
        if self.is_loc: self.pose_net = build_decoder(args)
        
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
    def __init__(self, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
        """
        super().__init__()
        
        self.num_classes = 1
        
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses      
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)
        
        if "retrieval" in losses:          
            self.soft_triplet_loss = SoftTripletBiLoss()
        
    def loss_retrieval(self, outputs, targets, indices):        
        loss_ir, mean_p, mean_n = self.soft_triplet_loss(outputs["grnd"], outputs["arl"])
        losses = {"retrieval": loss_ir}
        return losses
    
    def loss_labels(self, outputs, targets, indices):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"labels": loss_ce}

        return losses
        
    def loss_boxes(self, outputs, targets, indices):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

        losses = {}
        losses["boxes"] = loss_bbox.sum() 

        return losses
    
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
        
    def get_loss(self, loss, outputs, targets, indices):
        loss_map = {
            "retrieval": self.loss_retrieval,
            "labels": self.loss_labels,
            "boxes": self.loss_boxes,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss" doc
        """  
        if "labels" in self.losses:      
            indices = self.matcher(outputs, targets)
        else:
            indices = None
        
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices))
            
        for k in losses.keys():
            losses[k] = losses[k] * self.weight_dict[k]

        return losses    
    
class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"] # bs x num_quries x 4
        
        assert len(out_logits) == len(targets)
        
        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)
        
        x_c, y_c, c, s = out_bbox.unbind(-1) # bs x num_quries
        yaw = torch.atan2(s, c)       
        
        xs, ys = [], []        
        for b in range(len(out_logits)):
            arl_img_size = targets[b]["orig_size"]
            arl_zoom_ratio = targets[b]["arl_zoom_ratio"][0]
            meter_per_pixel = targets[b]["meter_per_pixel"][0]
            # shift_range_lon = targets[b]["shift_range_lon"][0]
            # shift_range_lat = targets[b]["shift_range_lat"][0]            
            x = x_c[b] * arl_img_size[0] * arl_zoom_ratio * meter_per_pixel
            y = y_c[b] * arl_img_size[1] * arl_zoom_ratio * meter_per_pixel           
            xs.append(x)
            ys.append(y)
        xs = torch.stack(xs, 0)
        ys = torch.stack(ys, 0)  
         
        boxes = torch.stack([xs, ys, yaw], dim=-1)

        results = [{"scores": s, "labels": l, "boxes": b} for s, l, b in zip(scores, labels, boxes)]
        
        return results
    
def build(args, is_local, device):
    # build model
    model = CITRUS(args, is_local).to(device)
    
    # build criterion    
    if is_local:
        matcher = build_matcher(args)
        weight_dict = {"retrieval": 1, "labels": args["label_loss_coef"], "boxes": args["bbox_loss_coef"]}
        eos_coef = args["eos_coef"]
        losses = ["retrieval", "labels", "boxes"]   
    else:
        matcher = None
        weight_dict = {"retrieval": 1}
        eos_coef = 0
        losses = ["retrieval"]
    criterion = SetCriterion(matcher=matcher, weight_dict=weight_dict, eos_coef=eos_coef, losses=losses).to(device)
    
    # build post processor   
    if is_local:    
        postprocessors = {"bbox": PostProcess()}
    else:
        postprocessors = None
    
    return model, criterion, postprocessors