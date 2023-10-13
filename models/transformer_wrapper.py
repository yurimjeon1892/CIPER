import torch
import torch.nn.functional as F

from torch import nn
from functools import partial
from timm.models.vision_transformer import VisionTransformer

import torchvision
import numpy as np

from .transformer import TransformerDecoder, TransformerDecoderLayer

class Encoder(VisionTransformer):
    
    def __init__(self, args, img_size, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__(img_size=img_size, patch_size=args["patch_size"], embed_dim=args["dim_embed"], num_classes=args["dim_feature"], depth=args["num_enc_layers"], num_heads=args["num_heads"], mlp_ratio=args["mlp_ratio"], qkv_bias=args["qkv_bias"], norm_layer=norm_layer)

        num_patches = self.patch_embed.num_patches
        
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))          
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim)) 
        
        nn.init.trunc_normal_(self.cls_token)        
        nn.init.trunc_normal_(self.dist_token)
        
        self.head = nn.Linear(self.embed_dim, self.num_classes)
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) 
        
        self.head.apply(self._init_weights)
        self.head_dist.apply(self._init_weights)
                
        self._load_pretrained(img_size, args["dim_feature"])
        
    def _load_pretrained(self, img_size, num_classes):        
        checkpoint = torch.hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth", map_location="cpu")     
           
        weight = checkpoint["model"]['pos_embed']
        ori_size = np.sqrt(weight.shape[1] - 1).astype(int)
        new_size = (img_size[0] // self.patch_embed.patch_size[0], img_size[1] // self.patch_embed.patch_size[1])
        matrix = weight[:, 2:, :].reshape([1, ori_size, ori_size, weight.shape[-1]]).permute((0, 3, 1, 2))
        resize = torchvision.transforms.Resize(new_size)
        new_matrix = resize(matrix).permute(0, 2, 3, 1).reshape([1, -1, weight.shape[-1]])
        checkpoint["model"]['pos_embed'] = torch.cat([weight[:, :2, :], new_matrix], dim=1)
        # change the prediction head if not 1000
        if num_classes != 1000:
            checkpoint["model"]['head.weight'] = checkpoint["model"]['head.weight'].repeat(5,1)[:num_classes, :]
            checkpoint["model"]['head.bias'] = checkpoint["model"]['head.bias'].repeat(5)[:num_classes]
            checkpoint["model"]['head_dist.weight'] = checkpoint["model"]['head.weight'].repeat(5,1)[:num_classes, :]
            checkpoint["model"]['head_dist.bias'] = checkpoint["model"]['head.bias'].repeat(5)[:num_classes]
        msg = self.load_state_dict(checkpoint["model"])
        print(msg)
    
    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            x = blk(x)

        x = self.norm(x)
        return x[:, 0], x[:, 1], x[:, 2:], self.pos_embed[:, 2:]
    
    def forward(self, x):
        x, x_dist, mem, pos = self.forward_features(x)
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        # follow the evaluation of deit, simple average and no distillation during training, could remove the x_dist
        return (x + x_dist) / 2, (mem, pos)
    
class Decoder(nn.Module):
    def __init__(self, args):
        """ Initializes the model.
        Parameters:
            dim_embed: size of the embeddings (dimension of the transformer)
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
        """
        super().__init__()
        
        num_patches = int(args["grd_img_size"][0] / args["patch_size"]) * int(args["grd_img_size"][1] / args["patch_size"])  
        dim_embed_dec = args["dim_embed"] + args["dim_embed_2"]
        
        self.query_embed = nn.Embedding(args["num_queries"], dim_embed_dec)         
        self.class_embed = nn.Linear(args["dim_embed"]  + args["dim_embed_2"], 2)      
        self.bbox_embed = MLP(args["dim_embed"] + args["dim_embed_2"], args["dim_embed"] + args["dim_embed_2"], output_dim=4, num_layers=3)
        
        decoder_layer = TransformerDecoderLayer(d_model=dim_embed_dec, 
                                                nhead=args["num_heads"], 
                                                dim_feedforward=args["dim_feedforward"],
                                                dropout=args["dropout"], 
                                                activation="relu", 
                                                normalize_before=args["pre_norm"])
        decoder_norm = nn.LayerNorm(dim_embed_dec)
        self.decoder = TransformerDecoder(decoder_layer, args["num_dec_layers"], decoder_norm,
                                          return_intermediate=False)
            
        self.mlp_mem_grd = MLP(
            input_dim=args["dim_embed"],
            hidden_dim=args["dim_embed"],
            output_dim=1,
            num_layers=args["mlp_ratio"],
        )        
        self.lin_mem_grd = nn.Linear(num_patches, args["dim_embed_2"])         
        self.mlp_pos_grd = MLP(
            input_dim=args["dim_embed"],
            hidden_dim=args["dim_embed"],
            output_dim=1,
            num_layers=args["mlp_ratio"],
        )
        self.lin_pos_grd = nn.Linear(num_patches, args["dim_embed_2"])  
        
    def forward(self, x_grd, x_arl):   
        """
            memory_grd: bs x num_patches1 x dim_embed
            pos_grd: 1 x num_patches1 x dim_embed
            memory_arl: bs x num_patches2 x dim_embed
            pos_arl: 1 x num_patches2 x dim_embed
        """
                
        memory_grd, pos_grd = x_grd[0], x_grd[1]
        memory_arl, pos_arl = x_arl[0], x_arl[1]
        bs = memory_arl.size(0)
        # print("in: ", memory_grd.size(), pos_grd.size(), memory_arl.size(), pos_arl.size())
             
        memory_grd = self.mlp_mem_grd(memory_grd) # bs x num_patches1 x 1
        # print("1a: ", memory_grd.size())
        memory_grd = self.lin_mem_grd(memory_grd.squeeze(2)) # bs x dim_embed_dec
        # print("2a: ", memory_grd.size())
        memory_grd = memory_grd.unsqueeze(1).repeat(1, memory_arl.size(1), 1) # bs x num_patches2 x dim_embed_dec
        # print("3a: ", memory_grd.size())
        
        memory = torch.cat([memory_grd, memory_arl], 2).permute(1, 0, 2) # num_patches2 x bs x dim_embed_dec
        # print("mem: ", memory_grd.size(), memory_arl.size(), memory.size())
                
        pos_grd = self.mlp_pos_grd(pos_grd) # bs x num_patches1 x 1
        # print("1b: ", pos_grd.size())
        pos_grd = self.lin_pos_grd(pos_grd.squeeze(2)) # bs x dim_embed_dec
        # print("2b: ", pos_grd.size())
        pos_grd = pos_grd.unsqueeze(1).repeat(1, pos_arl.size(1), 1) # bs x num_patches2 x dim_embed_dec
        # print("3b: ", pos_grd.size())
          
        pos_embed = torch.cat([pos_grd, pos_arl], 2).permute(1, 0, 2) # num_patches2 x bs x dim_embed_dec
        # print("pos: ", pos_grd.size(), pos_arl.size(), pos_embed.size())
            
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1) # num_queries x bs x dim_embed_dec
        # print("query_embed: ", query_embed.size())
        
        # print("input", memory.size(), pos_embed.size(), query_embed.size())
        tgt = torch.zeros_like(query_embed) # output 이 저장되는 공간         
        dst = self.decoder(tgt, memory, 
                           pos=pos_embed, query_pos=query_embed
                         ) 
        dst = dst.transpose(1, 2) # 1 x bs x num_quries x dim_embed_dec
        # print("dst: ", dst.size())
        
        outputs_class = self.class_embed(dst) # 1 x bs x num_quries x 2
        outputs_coord = self.bbox_embed(dst) # 1 x bs x num_quries x 4   
        # print("out: ", outputs_class.size(), outputs_coord.size())     
        
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]} # [-1]: 가장 마지막 decoder layer결과만 사용
        
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

def build_transformer_decoder(args):
    return Decoder(args)