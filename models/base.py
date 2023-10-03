# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn.functional as F
# import math
# from scipy.stats import truncnorm
from torch import nn

from common.utils_misc import nested_tensor_from_tensor_list, NestedTensor

from .backbone import build_backbone
from .transformer_wrapper import build_transformer_encoder, build_transformer_decoder

class Base1(nn.Module):
    """ This is the base ciper module that performs cross-view image geo-localization """
    def __init__(self, args, backbone, transformer):
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
        # self.num_queries = num_queries
        self.transformer = transformer
        # hidden_dim = transformer.d_model
        # self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        # self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        # self.query_embed = nn.Embedding(num_queries, hidden_dim)
        # self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.input_proj = nn.Conv2d(backbone.num_channels, args["dim_embed"], kernel_size=1)
        self.backbone = backbone
        # self.aux_loss = aux_loss
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, args["dim_embed"]))          
        # self.dist_token = nn.Parameter(torch.zeros(1, 1, args["dim_embed"])) 
        # self.pos_embed = nn.Parameter(torch.zeros(1, 1, args["dim_embed"]))  
        
        self.head = nn.Linear(args["dim_embed"], args["dim_feature"])
        # self.head_dist = nn.Linear(args["dim_embed"], args["dim_feature"]) 
        
        nn.init.trunc_normal_(self.cls_token)        
        # nn.init.trunc_normal_(self.dist_token)
        # nn.init.trunc_normal_(self.pos_embed)    
        
    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose() # bs x dim_embed x h x w, bs x h x w
        assert mask is not None
        # hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]
        src = self.input_proj(src)  
        
        #### inside the transformer    
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1) # num_patches(=h*w) x bs x dim_embed       
        # pos_embed = pos_embed.flatten(2).permute(2, 0, 1) # num_patches x bs x dim_embed   
        # query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        # mask = mask.flatten(1) # bs x num_patches      
          
        cls_token = self.cls_token.expand(bs, -1, -1).permute(1, 0, 2) # 1 x bs x dim_embed
        # dist_token = self.dist_token.expand(bs, -1, -1).permute(1, 0, 2) # 1 x bs x dim_embed
        src = torch.cat((cls_token, src), dim=0) # (num_patches + 1) x bs x dim_embed
                            
        # mask_ = self.cls_mask.expand(bs, -1)
        # mask = torch.cat((mask_, mask), dim=1) # bs x (num_patches + 1)
        
        memory = self.transformer(src, None, None, None)  # (num_patches + 1) x bs x dim_embed
        
        ### outside the transformer
        memory = memory.permute(1, 2, 0) # bs x dim_embed x (num_patches + 1)        
        emb = self.head(memory[:, :, 0]) # bs x dim_embed             
        mem = memory[:, :, 1:].view(bs, c, h, w) # bs x dim_embed x h x w
        
        return emb, mem, mask, pos[-1]

    
def build_base1(args):
    
    backbone = build_backbone(args)
    
    transformer = build_transformer_encoder(args)
    
    base1 = Base1(
        args,
        backbone,
        transformer,
        # num_classes=num_classes,
        # num_queries=args.num_queries,
        # aux_loss=args.aux_loss,
    )
        
    return base1

def build_base2(args, IS_POSE):
    if not IS_POSE: return None