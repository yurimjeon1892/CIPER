import torch
# import torch.nn.functional as F
from torch import nn

import math

from common.utils_misc import nested_tensor_from_tensor_list

from .backbone import build_backbone
from .transformer import build_transformer_encoder, build_transformer_decoder

class CITRENC(nn.Module):
    """ This is the Cigarette module that performs cross-view image geo-localization """
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
                
        self.backbone = build_backbone(args)       
        self.encoder = build_transformer_encoder(args) 
        
        # self._reset_parameters()
        self.input_proj = nn.Conv2d(self.backbone.num_channels, args["hidden_dim"], kernel_size=1) 
        
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)   

    def forward(self, x):
        if isinstance(x, (list, torch.Tensor)):
            x = nested_tensor_from_tensor_list(x)     
        
        features, pos = self.backbone(x)
        src, mask = features[-1].decompose() # torch.Size([2, 512, 8, 32]) torch.Size([2, 8, 32])
        assert mask is not None
        out_emb, memory = self.encoder(self.input_proj(src), mask, pos[-1])    
        
        return out_emb, memory
    
def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.

    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)

def pos2embed(pos, num_pos_feats=128):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = 2 * (dim_t // 2) / num_pos_feats + 1
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x), dim=-1)
    return posemb
    
class CITRDEC(nn.Module):
    """ This is the Cigarette module that performs cross-view image geo-localization """
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
        self.hidden_dim = args["hidden_dim"]
        num_queries = args["num_queries"]
        
        self.grnd_img_size = args["grnd_img_size"]        
        self.arl_img_size = args["arl_img_size"]
        self.down_ratio = args["down_ratio"]
        
        self.device = args["device"]
        self.batch_size = args["batch_size"]
        
        self.reference_points = nn.Embedding(num_queries, 3)
        self.embedding_arl = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim ),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim , self.hidden_dim )
        )
        self.embedding_grnd = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim ),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim , self.hidden_dim )
        )
        
        self.task_class = nn.Sequential(
            nn.Conv1d(self.hidden_dim, self.hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_dim, 2, 1),
            nn.Softmax()
        )
        
        self.task_bbox = nn.Sequential(
            nn.Conv1d(self.hidden_dim, self.hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_dim, 3, 1)
        )        
        
        self.decoder = build_transformer_decoder(args)
        
        self._reset_parameters()
        self._init_weights()
        
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)        
        
    def _init_weights(self):
        nn.init.uniform_(self.reference_points.weight.data, 0, 1)    
        
    @property
    def coords_arl(self):
        x_size, y_size = (
            self.arl_img_size[1] // self.down_ratio,
            self.arl_img_size[0] // self.down_ratio
        )
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        batch_y, batch_x = torch.meshgrid(*[torch.linspace(it[0], it[1], it[2]) for it in meshgrid])
        batch_x = (batch_x + 0.5) / x_size
        batch_y = (batch_y + 0.5) / y_size
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)
        coord_base = coord_base.view(2, -1).transpose(1, 0) # (H*W, 2)
        return coord_base
    
    @property
    def coords_grnd(self):
        x_size, y_size = (
            self.grnd_img_size[1] // self.down_ratio,
            self.grnd_img_size[0] // self.down_ratio
        )
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        batch_y, batch_x = torch.meshgrid(*[torch.linspace(it[0], it[1], it[2]) for it in meshgrid])
        batch_x = (batch_x + 0.5) / x_size
        batch_y = (batch_y + 0.5) / y_size
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)
        coord_base = coord_base.view(2, -1).transpose(1, 0) # (H*W, 2)
        return coord_base
    
    def _arl_img_query_embed(self, ref_points):
        arl_img_embeds = self.embedding_arl(pos2embed(ref_points, num_pos_feats=self.hidden_dim))
        return arl_img_embeds
    
    def query_embed(self, ref_points):
        ref_points = inverse_sigmoid(ref_points.clone()).sigmoid()
        arl_img_embeds = self._arl_img_query_embed(ref_points)
        return arl_img_embeds

    def forward(self, x_grnd, x_arl):        
        
        reference_points = self.reference_points.weight
        reference_points = reference_points.unsqueeze(0).repeat(self.batch_size, 1, 1)
        attn_mask = None       
        
        pos_embed_arl = self.embedding_arl(pos2embed(self.coords_arl.to(x_arl.device), num_pos_feats=self.hidden_dim))
        pos_embed_grnd = self.embedding_grnd(pos2embed(self.coords_grnd.to(x_grnd.device), num_pos_feats=self.hidden_dim))        
        
        query_embeds = self.query_embed(reference_points)
        
        outs_dec = self.decoder(
                            x_arl, x_grnd, query_embeds,
                            pos_embed_arl, pos_embed_grnd,
                            attn_masks=attn_mask
                        ) # [2, 128, 100]
        outs_dec = torch.nan_to_num(outs_dec)
        
        outputs_class = self.task_class(outs_dec)
        outputs_class = outputs_class.permute(0, 2, 1) # [2, 100, 2]
        
        outputs_coord = self.task_bbox(outs_dec)
        outputs_coord = outputs_coord.permute(0, 2, 1) # [2, 100, 3]

        reference = inverse_sigmoid(reference_points.clone())
        outputs_coord = (outputs_coord + reference).sigmoid()
        # center = (outs + reference[None, :, :, :2]).sigmoid()
        # _center = center[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
                
        # outputs_class = self.class_embed(hs) # ([6, 2 batch, 100개 object, 92개 class])
        # outputs_coord = self.bbox_embed(hs).sigmoid() # ([6, 2, 100, 4])
        
        out = {'pred_logits': outputs_class, 'pred_boxes': outputs_coord} # [-1]: 가장 마지막 decoder layer결과만 사용
        # (bs, query_num, 2), (bs, query_num, 3) 
        
        return out

def build_encoder(args):
    return CITRENC(args)

def build_decoder(args):
    return CITRDEC(args)