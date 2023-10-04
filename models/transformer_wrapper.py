import torch
import torch.nn.functional as F
from torch import nn
from functools import partial

from timm.models.layers import PatchEmbed

from common.utils_misc import nested_tensor_from_tensor_list, NestedTensor
from .transformer import TransformerEncoderLayer, TransformerEncoder
from .backbone import build_backbone


from .vision_transformer import Block, Mlp
from timm.models.vision_transformer import VisionTransformer, init_weights_vit_timm, get_init_weights_vit, named_apply
from timm.models.layers import trunc_normal_

# class EncoderWrapper(VisionTransformer):
    
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         # self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

#         num_patches = self.patch_embed.num_patches
#         self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
#         # self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

#         # trunc_normal_(self.dist_token, std=.02)
#         # trunc_normal_(self.pos_embed, std=.02)
#         # self.head_dist.apply(self._init_weights)
        
#         self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))          
#         self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim)) 
        
#         self.head = nn.Linear(self.embed_dim, self.num_classes)
#         self.head_dist = nn.Linear(self.embed_dim, self.num_classes) 
        
#         nn.init.trunc_normal_(self.cls_token)        
#         nn.init.trunc_normal_(self.dist_token)
        
#         self.head.apply(self._init_weights)
#         self.head_dist.apply(self._init_weights)

#     def forward(self, x):
#         # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
#         # with slight modifications to add the dist_token
#         B = x.shape[0]
#         x = self.patch_embed(x) # [8, 3, 320, 320] --> [8, 400, 384]
          
#         cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks # [8, 1, 384]
#         dist_token = self.dist_token.expand(B, -1, -1) # [8, 1, 384]
#         x = torch.cat((cls_tokens, dist_token, x), dim=1) # [8, 402, 384]

#         x = x + self.pos_embed # [8, 402, 384]
#         x = self.pos_drop(x)
                            
#         for i, blk in enumerate(self.blocks):
#             x = blk(x)

#         x = self.norm(x) # [8, 402, 384]
#         x, x_dist = x[:, 0], x[:, 1]
        
#         x = self.head(x) # bs x dim_embed      
#         x_dist =  self.head_dist(x_dist)
#         return (x + x_dist) / 2

# class EncoderWrapper(nn.Module):

#     def __init__(self, args, img_size):
#         super().__init__()

#         encoder_layer = TransformerEncoderLayer(d_model=args["dim_embed"],
#                                                 nhead=args["num_heads"],
#                                                 dim_feedforward=args["dim_feedforward"],
#                                                 dropout=args["dropout"],                                                
#                                                 activation="relu",
#                                                 normalize_before=args["pre_norm"])
#         encoder_norm = nn.LayerNorm(args["dim_embed"]) if args["pre_norm"] else None
#         self.encoder = TransformerEncoder(encoder_layer, args["num_enc_layers"], encoder_norm) 
        
#         self.cls_token = nn.Parameter(torch.zeros(1, 1, args["dim_embed"]))          
#         self.dist_token = nn.Parameter(torch.zeros(1, 1, args["dim_embed"])) 
#         self.pos_embed = nn.Parameter(torch.zeros(1, 1, args["dim_embed"]))  
#         self.init_weights("")
        
#         self.head = nn.Linear(args["dim_embed"], args["dim_feature"])
#         self.head_dist = nn.Linear(args["dim_embed"], args["dim_feature"]) 
        
#         nn.init.trunc_normal_(self.cls_token)        
#         nn.init.trunc_normal_(self.dist_token)
#         # nn.init.trunc_normal_(self.pos_embed)   
                
#         patch_size = 16
#         in_chans = 3
#         embed_dim = args["dim_embed"]
#         pre_norm = False
#         # dynamic_img_pad = False        
#         self.patch_embed = PatchEmbed(
# 			img_size=img_size,
# 			patch_size=patch_size,
#             in_chans=in_chans,
#             embed_dim=embed_dim,
#             bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
#             # dynamic_img_pad=dynamic_img_pad,
# 		)

#         self.head.apply(self._init_weights)
#         self.head_dist.apply(self._init_weights)
        
#     def init_weights(self, mode=''):
#         import math
#         assert mode in ('jax', 'jax_nlhb', 'moco', '')
#         head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
#         # trunc_normal_(self.pos_embed, std=.02)
#         if self.cls_token is not None:
#             nn.init.normal_(self.cls_token, std=1e-6)
#         named_apply(get_init_weights_vit(mode, head_bias), self)
                
#     def _init_weights(self, m):
#         # this fn left here for compat with downstream users
#         init_weights_vit_timm(m)

#     def forward(self, x, mask=None, query_embed=None, pos_embed=None):
        
#         # flatten NxCxHxW to HWxNxC
#         bs = x.shape[0]
#         x = self.patch_embed(x)     
          
#         cls_token = self.cls_token.expand(bs, -1, -1) # 1 x bs x dim_embed
#         dist_token = self.dist_token.expand(bs, -1, -1) # 1 x bs x dim_embed
#         x = torch.cat((cls_token, dist_token, x), dim=1) # (num_patches + 1) x bs x dim_embed
                            
#         # mask_ = self.cls_mask.expand(bs, -1)
#         # mask = torch.cat((mask_, mask), dim=1) # bs x (num_patches + 1)

#         # dst = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed) # (num_patches + 1) x bs x dim_embed
#         # output = self.encoder(src, mask, query_embed, pos_embed)
        
#         x = x.permute(1, 0, 2)
#         x = self.encoder(x)
#         x = x.permute(1, 0, 2)
                    
#         x, x_dist = x[:, 0], x[:, 1]
        
#         x = self.head(x) # bs x dim_embed      
#         x_dist =  self.head_dist(x_dist)
#         emb = (x + x_dist) / 2
        
#         return emb

class EncoderWrapper(nn.Module):

    def __init__(self, args):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model=args["dim_embed"],
                                                nhead=args["num_heads"],
                                                dim_feedforward=args["dim_feedforward"],
                                                dropout=args["dropout"],                                                
                                                activation="relu",
                                                normalize_before=args["pre_norm"])
        encoder_norm = nn.LayerNorm(args["dim_embed"]) if args["pre_norm"] else None
        self.encoder = TransformerEncoder(encoder_layer, args["num_enc_layers"], encoder_norm) 
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, args["dim_embed"]))          
        self.dist_token = nn.Parameter(torch.zeros(1, 1, args["dim_embed"])) 
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, args["dim_embed"]))  
        self.init_weights("")
        
        self.head = nn.Linear(args["dim_embed"], args["dim_feature"])
        self.head_dist = nn.Linear(args["dim_embed"], args["dim_feature"]) 
        
        nn.init.trunc_normal_(self.cls_token)        
        nn.init.trunc_normal_(self.dist_token)
        # nn.init.trunc_normal_(self.pos_embed)   
                
        self.head.apply(self._init_weights)
        self.head_dist.apply(self._init_weights)
        
    def init_weights(self, mode=''):
        import math
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        # trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(get_init_weights_vit(mode, head_bias), self)
                
    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        init_weights_vit_timm(m)

    def forward(self, x, mask=None, query_embed=None, pos_embed=None):
        
        bs = x.shape[0]        
        x = x.flatten(2).permute(0, 2, 1)
          
        cls_token = self.cls_token.expand(bs, -1, -1) 
        dist_token = self.dist_token.expand(bs, -1, -1) 
        x = torch.cat((cls_token, dist_token, x), dim=1) 
        
        x = x.permute(1, 0, 2)
        x = self.encoder(x)
        x = x.permute(1, 0, 2)
                    
        x, x_dist = x[:, 0], x[:, 1]
        
        x = self.head(x) # bs x dim_embed      
        x_dist =  self.head_dist(x_dist)
        emb = (x + x_dist) / 2
        
        return emb
    
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

# def build_transformer_encoder(args, img_size):
#     return EncoderWrapper(args, img_size)
def build_transformer_encoder(args):
    return EncoderWrapper(args)

def build_transformer_decoder(args):
    return Decoder(args)