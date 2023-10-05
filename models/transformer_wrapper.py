import torch
import torch.nn.functional as F

from torch import nn
from functools import partial
from timm.models.vision_transformer import VisionTransformer

import torchvision
import numpy as np

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

    def forward(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B = x.shape[0]
        x = self.patch_embed(x) # [8, 3, 320, 320] --> [8, 400, 384]
          
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks # [8, 1, 384]
        dist_token = self.dist_token.expand(B, -1, -1) # [8, 1, 384]
        x = torch.cat((cls_tokens, dist_token, x), dim=1) # [8, 402, 384]

        x = x + self.pos_embed # [8, 402, 384]
        x = self.pos_drop(x)
                            
        for i, blk in enumerate(self.blocks):
            x = blk(x)

        x = self.norm(x) # [8, 402, 384]
        x, x_dist = x[:, 0], x[:, 1]
        
        x = self.head(x) # bs x dim_embed      
        x_dist =  self.head_dist(x_dist)
        return (x + x_dist) / 2
    
class Decoder(nn.Module):
    def __init__(self, args):
        """ Initializes the model.
        Parameters:
            dim_embed: size of the embeddings (dimension of the transformer)
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
        """
        super().__init__()
        
        # self.decoder = build_transformer_decoder(args)
        
        dim_embed_encoder = args["dim_embed"]
        dim_embed_decoder = args["dim_embed"] + int(dim_embed_encoder / 2)
        
        self.query_embed = nn.Embedding(args["num_queries"], dim_embed_decoder) 
        
        self.conv_mem_1 = nn.Sequential(
            nn.Conv2d(dim_embed_encoder, int(dim_embed_encoder / 4), 3, stride=(2, 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(dim_embed_encoder / 4), int(dim_embed_encoder / 16), 3, stride=(2, 2)),
            nn.ReLU(inplace=True),
        )
        self.conv_mem_2 = nn.Linear(int(dim_embed_encoder / 16), int(dim_embed_encoder / 2))  
        
        self.conv_pos_1 = nn.Sequential(
            nn.Conv2d(dim_embed_encoder, int(dim_embed_encoder / 4), 2, stride=(2, 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(dim_embed_encoder / 4), int(dim_embed_encoder / 16), 2, stride=(2, 2)),
            nn.ReLU(inplace=True),
        )
        self.conv_pos_2 = nn.Linear(int(dim_embed_encoder / 16), int(dim_embed_encoder / 2))  
        
        self.class_embed = nn.Linear(dim_embed_decoder, 2)      
        self.bbox_embed = MLP(dim_embed_decoder, dim_embed_decoder, output_dim=4, num_layers=3)
        
        decoder_layer = TransformerDecoderLayer(dim_embed=dim_embed_decoder, 
                                                num_heads=args["num_heads"], 
                                                dim_feedforward=args["dim_feedforward"],
                                                dropout=args["dropout"], 
                                                activation="relu", 
                                                normalize_before=args["pre_norm"])
        decoder_norm = nn.LayerNorm(dim_embed_decoder)
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
        # print("in: ", memory_grnd.size(), pos_grnd.size(), memory_arl.size(), pos_arl.size())
        
        memory_arl = memory_arl.flatten(2)        
        memory_grnd_feat = self.conv_mem_1(memory_grnd)        
                
        _, kc, kh, kw = memory_grnd_feat.size()  
        memory_filter = torch.randn(kc, kc, kh, kw).cuda()
        memory_grnd_feat = F.conv2d(memory_grnd_feat, memory_filter, padding=0)     
        
        memory_grnd_feat = self.conv_mem_2(memory_grnd_feat.view(bs, -1))        
        memory_grnd_feat = memory_grnd_feat.unsqueeze(-1).repeat(1, 1, memory_arl.size(2))        
        memory = torch.cat([memory_grnd_feat, memory_arl], 1).permute(2, 0, 1) # num_queries x bs x dim_embed
        # print("mem: ", memory_grnd_feat.size(), memory_arl.size())
                
        pos_arl = pos_arl.flatten(2)
        pos_grnd_feat = self.conv_pos_1(pos_grnd)
        
        _, kc, kh, kw = pos_grnd_feat.size()  
        pos_filter = torch.randn(kc, kc, kh, kw).cuda()
        pos_grnd_feat = F.conv2d(pos_grnd_feat, pos_filter, padding=0)     
        
        pos_grnd_feat = self.conv_pos_2(pos_grnd_feat.view(bs, -1))        
        pos_grnd_feat = pos_grnd_feat.unsqueeze(-1).repeat(1, 1, pos_arl.size(2))    
        pos_embed = torch.cat([pos_grnd_feat, pos_arl], 1).permute(2, 0, 1) # num_queries x bs x dim_embed  
        # print("pos: ", pos_grnd_feat.size(), pos_arl.size())
        
        mask = mask.flatten(1) # bs x num_queries        
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1) # num_queries x bs x dim_embed
        
        # print("input", memory.size(), mask.size(), pos_embed.size(), query_embed.size())
        tgt = torch.zeros_like(query_embed) # output 이 저장되는 공간         
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

def build_transformer_decoder(args):
    return Decoder(args)