import torch
import torch.nn.functional as F
import math
from torch import nn
from scipy.stats import truncnorm

from common.utils_misc import nested_tensor_from_tensor_list

from .backbone import build_backbone
from .transformer import *

class CIPERENC(nn.Module):
    def __init__(self, args):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
        """
        super().__init__()
                
        self.backbone = build_backbone(args)        
        self.input_proj = nn.Conv2d(self.backbone.num_channels, args["dim_embed"], kernel_size=1) 
        
        encoder_layer = TransformerEncoderLayer(dim_embed=args["dim_embed"],
                                                num_heads=args["num_heads"],
                                                dim_feedforward=args["dim_feedforward"],
                                                dropout=args["dropout"],                                                
                                                activation="relu",
                                                normalize_before=args["pre_norm"])
        encoder_norm = nn.LayerNorm(args["dim_embed"]) if args["pre_norm"] else None
        self.encoder = TransformerEncoder(encoder_layer, args["num_enc_layers"], encoder_norm)   
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, args["dim_embed"]))          
        self.dist_token = nn.Parameter(torch.zeros(1, 1, args["dim_embed"])) 
        self.pos_embed = nn.Parameter(torch.zeros(1, 2, args["dim_embed"]))  
        
        self.head = nn.Linear(args["dim_embed"], args["dim_feature"])
        self.head_dist = nn.Linear(args["dim_embed"], args["dim_feature"]) 

    def forward(self, x):
        if isinstance(x, (list, torch.Tensor)):
            x = nested_tensor_from_tensor_list(x)     
        
        features, pos = self.backbone(x)
        src, mask = features[-1].decompose() # bs x dim_embed x h x w, bs x h x w
        assert mask is not None
        # embed, memory = self.encoder(self.input_proj(src), mask, pos[-1])  
        
        src = self.input_proj(src)        
        pos_embed = pos[-1]       
         
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape 
        
        cls_token = self.cls_token.expand(bs, -1, -1).permute(1, 0, 2) # 1 x bs x dim_embed
        dist_token = self.dist_token.expand(bs, -1, -1).permute(1, 0, 2) # 1 x bs x dim_embed
        src = src.flatten(2).permute(2, 0, 1) # num_patches(=h*w) x bs x dim_embed
        src = torch.cat((cls_token, dist_token, src), dim=0) # (num_patches + 1) x bs x dim_embed
        
        # mask = mask.flatten(1) # bs x num_patches        
        # mask_ = self.cls_mask.expand(bs, -1)
        # mask = torch.cat((mask_, mask), dim=1) # bs x (num_patches + 1)
        
        pos_embed_ = self.pos_embed.expand(bs, -1, -1).permute(1, 0, 2) # 1 x bs x dim_embed        
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1) # num_patches x bs x dim_embed    
        pos_embed = torch.cat((pos_embed_, pos_embed), dim=0) # (num_patches + 1) x bs x dim_embed

        # dst = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed) # (num_patches + 1) x bs x dim_embed
        dst = self.encoder(src, pos=pos_embed) # (num_patches + 1) x bs x dim_embed
        dst = dst.permute(1, 2, 0) # bs x dim_embed x (num_patches + 1)
        
        x = self.head(dst[:, :, 0]) # bs x dim_embed     
        x_dist = self.head_dist(dst[:, :, 1]) # bs x dim_embed     
        embed = (x + x_dist) / 2
        
        memory = dst[:, :, 2:].view(bs, c, h, w) # bs x dim_embed x h x w
        
        return embed, memory, mask, pos[-1]
    
class CIPERDEC(nn.Module):
    def __init__(self, args):
        """ Initializes the model.
        Parameters:
            dim_embed: size of the embeddings (dimension of the transformer)
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
        """
        super().__init__()
        
        # self.decoder = build_transformer_decoder(args)
        
        dim_embed = args["dim_embed"]
        dim_embed_merged = dim_embed + int(dim_embed / 2)
        
        self.query_embed = nn.Embedding(args["num_queries"], dim_embed_merged) 
        
        self.conv_mem = nn.Sequential(
            nn.Conv2d(dim_embed, int(dim_embed / 4), 2, stride=(1, 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(dim_embed / 4), int(dim_embed / 16), 2, stride=(1, 2)),
        )
        self.conv_pos = nn.Sequential(
            nn.Conv2d(dim_embed, int(dim_embed / 4), 2, stride=(1, 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(dim_embed / 4), int(dim_embed / 16), 2, stride=(1, 2)),
        )
        
        self.class_embed = nn.Linear(dim_embed_merged, 2)      
        self.bbox_embed = MLP(dim_embed_merged, dim_embed_merged, output_dim=4, num_layers=3)
        
        decoder_layer = TransformerDecoderLayer(dim_embed=dim_embed_merged, 
                                                num_heads=args["num_heads"], 
                                                dim_feedforward=args["dim_feedforward"],
                                                dropout=args["dropout"], 
                                                activation="relu", 
                                                normalize_before=args["pre_norm"])
        decoder_norm = nn.LayerNorm(dim_embed_merged)
        self.decoder = TransformerDecoder(decoder_layer, args["num_dec_layers"], decoder_norm,
                                          return_intermediate=False)
        
    def forward(self, x_grnd, x_arl, mask):   
        """
            memory_grnd: bs x dim_embed x h1 x w1
            pos_grnd: bs x dim_embed x h1 x w1
            memory_arl: bs x dim_embed x h2 x w2
            pos_arl: bs x dim_embed x h2 x w2
        """
                
        memory_grnd, pos_grnd = x_grnd[0], x_grnd[1]
        memory_arl, pos_arl = x_arl[0], x_arl[1]
        
        bs = memory_arl.size(0)
        
        memory_arl = memory_arl.flatten(2)        
        memory_grnd_feat = self.conv_mem(memory_grnd).view(bs, -1)
        memory_grnd_feat = memory_grnd_feat.unsqueeze(-1).repeat(1, 1, memory_arl.size(2))
        memory = torch.cat([memory_grnd_feat, memory_arl], 1).permute(2, 0, 1) # num_queries x bs x dim_embed
                
        pos_arl = pos_arl.flatten(2)
        pos_grnd_feat = self.conv_pos(pos_grnd).reshape(bs, -1)
        pos_grnd_feat = pos_grnd_feat.unsqueeze(-1).repeat(1, 1, pos_arl.size(2))
        pos_embed = torch.cat([pos_grnd_feat, pos_arl], 1).permute(2, 0, 1) # num_queries x bs x dim_embed
        
        mask = mask.flatten(1) # bs x num_queries        
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1) # num_queries x bs x dim_embed
        
        tgt = torch.zeros_like(query_embed) # output 이 저장되는 공간         
        # print("input", tgt.size(), memory.size(), mask.size(), pos_embed.size(), query_embed.size())
        dst = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                           pos=pos_embed, query_pos=query_embed
                         ) # num_quries x bs x dim_embed 
        dst = dst.transpose(0, 1) # bs x num_quries x dim_embed
        # print("dst: ", dst.size())
        
        outputs_class = self.class_embed(dst) # bs x num_quries x 2
        outputs_coord = self.bbox_embed(dst) # bs x num_quries x 4   
        # print("out: ", outputs_class.size(), outputs_coord.size())     
        
        out = {'pred_logits': outputs_class, 'pred_boxes': outputs_coord} # [-1]: 가장 마지막 decoder layer결과만 사용
        
        return out
    
class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def build_encoder(args):
    return CIPERENC(args)

def build_decoder(args):
    return CIPERDEC(args)