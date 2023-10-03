import torch
import torch.nn.functional as F
from torch import nn

from .transformer import *

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

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # # flatten NxCxHxW to HWxNxC
        # bs, c, h, w = src.shape
        # src = src.flatten(2).permute(2, 0, 1)
        # pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        # query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        # mask = mask.flatten(1)

        dst = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        
        return dst
    
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

def build_transformer_encoder(args):
    return EncoderWrapper(args)

def build_transformer_decoder(args):
    return Decoder(args)