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
                
        self._load_pretrained(img_size, args["dim_feature"], args["patch_size"])
        
    def _load_pretrained(self, img_size, num_classes, patch_size):        
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
        if patch_size != 16:
            checkpoint["model"]['patch_embed.proj.weight'] = checkpoint["model"]['patch_embed.proj.weight'].repeat(1, 1, int(patch_size / 16),int(patch_size / 16))
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
        return x[:, 0], x[:, 1], x[:, 2:]
    
    def forward(self, x):
        x, x_dist, mem = self.forward_features(x)
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        # follow the evaluation of deit, simple average and no distillation during training, could remove the x_dist
        return (x + x_dist) / 2, mem
    
class Decoder(nn.Module):
    def __init__(self, args):
        """ Initializes the model.
        Parameters:
            dim_embed: size of the embeddings (dimension of the transformer)
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
        """
        super().__init__()
        
        decoder_layer = TransformerDecoderLayer(d_model=args["dim_embed"], 
                                                nhead=args["num_heads"], 
                                                dim_feedforward=args["dim_feedforward"],
                                                dropout=args["dropout"], 
                                                activation="relu", 
                                                normalize_before=args["pre_norm"])
        decoder_norm = nn.LayerNorm(args["dim_embed"])
        self.decoder = TransformerDecoder(decoder_layer, args["num_dec_layers"], decoder_norm,
                                          return_intermediate=False)      
        
        self.query_conv = nn.Conv2d(args["dim_embed"], args["dim_embed"], kernel_size=2, stride=2)
        
        self.class_embed = nn.Linear(args["dim_embed"], 2)      
        self.bbox_embed = MLP(args["dim_embed"], args["dim_embed"] * 4, output_dim=4, num_layers=3)           
        
        num_queries = int((args["arl_img_size"][0] / args["patch_size"]) * (args["arl_img_size"][1] / args["patch_size"]) )
        self.query_embed = nn.Embedding(num_queries, args["dim_embed"]) 
        
    def forward(self, x_grd, x_arl):   
        """
            x_grd: bs x num_patches1 x dim_embed
            x_arl: bs x num_patches2 x dim_embed
        """
        
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, x_arl.size(0), 1)
        
        # q2 = int(x_arl.size(1) ** 0.5)
        # x_arl = x_arl.permute(0, 2, 1).view((x_arl.size(0), x_arl.size(2), q2, q2))
        # x_arl = self.query_conv(x_arl).flatten(2).permute(0, 2, 1)
        
        x_grd = x_grd.permute(1, 0, 2) # num_patches1 x bs x dim_embed
        x_arl = x_arl.permute(1, 0, 2) # num_patches2 x bs x dim_embed
        
        # print("input", x_grd.size(), x_arl.size(), query_embed.size())
        
        # tgt = torch.zeros_like(x_arl) # output 이 저장되는 공간         
        dst = self.decoder(x_arl, x_grd, 
                           query_pos=query_embed) 
        dst = dst.transpose(1, 2) # 1 x bs x num_patches2 x dim_embed_dec
        # print("dst: ", dst.size())
        
        outputs_class = self.class_embed(dst) # 1 x bs x num_patches2 x 2
        outputs_coord = self.bbox_embed(dst) # 1 x bs x num_patches2 x 4   
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