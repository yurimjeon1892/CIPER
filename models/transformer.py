# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

# class TransformerEncoderWrapper(nn.Module):

#     def __init__(self, dim_embed=512, num_heads=8, num_encoder_layers=6,
#                  dim_feedforward=2048, dropout=0.1,
#                  activation="relu", normalize_before=False):
#         super().__init__()
        
#         encoder_layer = TransformerEncoderLayer(dim_embed, num_heads, dim_feedforward,
#                                                 dropout, activation, normalize_before)
#         encoder_norm = nn.LayerNorm(dim_embed) if normalize_before else None
#         self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)   
        
#         self.cls_token = nn.Parameter(torch.zeros(1, 1, dim_embed))          
#         self.dist_token = nn.Parameter(torch.zeros(1, 1, dim_embed)) 
#         # self.cls_mask = nn.Parameter(torch.zeros(1, 1))           
#         self.pos_embed = nn.Parameter(torch.zeros(1, 2, dim_embed))  
        
#         self.head = nn.Linear(dim_embed, 1000)
#         self.head_dist = nn.Linear(dim_embed, 1000) 
        
#         self._reset_parameters()
        
#     def _reset_parameters(self):
#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)
#         nn.init.normal_(self.cls_token, std=1e-6)
#         nn.init.trunc_normal_(self.dist_token, std=.02)
#         nn.init.trunc_normal_(self.pos_embed, std=.02)
        
#     def forward(self, src, mask, pos_embed):
#         """
#             src: bs x dim_embed x h x w
#             mask: bs x h x w
#             pos_embed: bs x dim_embed x h x w
#         """
#         # flatten NxCxHxW to HWxNxC
#         bs, c, h, w = src.shape        
        
#         cls_token = self.cls_token.expand(bs, -1, -1).permute(1, 0, 2) # 1 x bs x dim_embed
#         dist_token = self.dist_token.expand(bs, -1, -1).permute(1, 0, 2) # 1 x bs x dim_embed
#         src = src.flatten(2).permute(2, 0, 1) # num_patches(=h*w) x bs x dim_embed
#         src = torch.cat((cls_token, dist_token, src), dim=0) # (num_patches + 1) x bs x dim_embed
        
#         # mask = mask.flatten(1) # bs x num_patches        
#         # mask_ = self.cls_mask.expand(bs, -1)
#         # mask = torch.cat((mask_, mask), dim=1) # bs x (num_patches + 1)
        
#         pos_embed_ = self.pos_embed.expand(bs, -1, -1).permute(1, 0, 2) # 1 x bs x dim_embed        
#         pos_embed = pos_embed.flatten(2).permute(2, 0, 1) # num_patches x bs x dim_embed    
#         pos_embed = torch.cat((pos_embed_, pos_embed), dim=0) # (num_patches + 1) x bs x dim_embed

#         # dst = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed) # (num_patches + 1) x bs x dim_embed
#         dst = self.encoder(src, pos=pos_embed) # (num_patches + 1) x bs x dim_embed
#         dst = dst.permute(1, 2, 0) # bs x dim_embed x (num_patches + 1)
        
#         x = self.head(dst[:, :, 0]) # bs x dim_embed     
#         x_dist = self.head_dist(dst[:, :, 1]) # bs x dim_embed     
#         embed = (x + x_dist) / 2
        
#         memory = dst[:, :, 2:].view(bs, c, h, w) # bs x dim_embed x h x w
        
#         return embed, memory 
    
# class TransformerDecoderWrapper(nn.Module):

#     def __init__(self, dim_embed=512, num_heads=8, 
#                  num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
#                  activation="relu", normalize_before=False,
#                  return_intermediate_dec=False):
#         super().__init__()
        
#         dim_embed = int(dim_embed * (3 / 2)) 

#         decoder_layer = TransformerDecoderLayer(dim_embed, num_heads, dim_feedforward,
#                                                 dropout, activation, normalize_before)
#         decoder_norm = nn.LayerNorm(dim_embed)
#         self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
#                                           return_intermediate=return_intermediate_dec)
        
#         self._reset_parameters()
        
#     def _reset_parameters(self):
#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)
    
#     def forward(self, tgt, memory,
#                 tgt_mask: Optional[Tensor] = None,
#                 memory_mask: Optional[Tensor] = None,
#                 tgt_key_padding_mask: Optional[Tensor] = None,
#                 memory_key_padding_mask: Optional[Tensor] = None,
#                 pos: Optional[Tensor] = None,
#                 query_pos: Optional[Tensor] = None):        
#         dst = self.decoder(
#             tgt, memory,
#             tgt_mask,
#             memory_mask,
#             tgt_key_padding_mask,
#             memory_key_padding_mask,
#             pos,
#             query_pos
#             )
#         return  dst
    

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output

class TransformerEncoderLayer(nn.Module):

    def __init__(self, dim_embed, num_heads, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim_embed, num_heads, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(dim_embed, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, dim_embed)

        self.norm1 = nn.LayerNorm(dim_embed)
        self.norm2 = nn.LayerNorm(dim_embed)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output

class TransformerDecoderLayer(nn.Module):

    def __init__(self, dim_embed, num_heads, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim_embed, num_heads, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(dim_embed, num_heads, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(dim_embed, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, dim_embed)

        self.norm1 = nn.LayerNorm(dim_embed)
        self.norm2 = nn.LayerNorm(dim_embed)
        self.norm3 = nn.LayerNorm(dim_embed)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

# def build_transformer_encoder(args):
#     return TransformerEncoderWrapper(
#         dim_embed=args["dim_embed"],
#         dropout=args["dropout"],
#         num_heads=args["num_heads"],
#         dim_feedforward=args["dim_feedforward"],
#         num_encoder_layers=args["num_enc_layers"],
#         normalize_before=args["pre_norm"],
#     )

# def build_transformer_decoder(args):
#     return TransformerDecoderWrapper(
#         dim_embed=args["dim_embed"],
#         dropout=args["dropout"],
#         num_heads=args["num_heads"],
#         dim_feedforward=args["dim_feedforward"],
#         num_decoder_layers=args["num_dec_layers"],
#         normalize_before=args["pre_norm"],
#     )