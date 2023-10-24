import torch
from torch import nn

from .common import MLP
from .transformer import TransformerDecoder, TransformerDecoderLayer
   
class Decoder(nn.Module):
    def __init__(self, args):
        """ Initializes the model.
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
                
        self.class_embed = nn.Linear(args["dim_embed"], 2)      
        self.bbox_embed = MLP(args["dim_embed"], args["dim_embed"] * 4, output_dim=4, num_layers=3)           
                
        self.num_queries = args["num_queries"]
        self.query_embed = nn.Embedding(self.num_queries, args["dim_embed"])    
        
    def forward(self, x_grd, x_arl):   
        """
            x_grd: bs x dim_embed
            x_arl: bs x num_patches2 x dim_embed
        """        
        x_grd = torch.repeat_interleave(x_grd.unsqueeze(0), self.num_queries, dim=0) # num_queries x bs x dim_embed          
        query_pos = torch.repeat_interleave(self.query_embed.weight.unsqueeze(1), x_grd.shape[1], dim=1) # num_queries x bs x dim_embed
              
        x_arl = x_arl.permute(1, 0, 2) # num_patches2 x bs x dim_embed
        # print("input", x_grd.size(), query_pos.size(), x_arl.size(), pos.size())
        
        dst = self.decoder(tgt=x_grd, memory=x_arl, 
                           query_pos=query_pos) 
        dst = dst.transpose(1, 2) # 1 x bs x num_patches2 x dim_embed_dec
        # print("dst: ", dst.size())
        
        outputs_class = self.class_embed(dst) # 1 x bs x num_patches2 x 2
        outputs_coord = self.bbox_embed(dst) # 1 x bs x num_patches2 x 4   
        # print("out: ", outputs_class.size(), outputs_coord.size())     
        
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]} # [-1]: 가장 마지막 decoder layer결과만 사용
        
        return out

# from typing import List, Tuple, Type
# from .common import LayerNorm2d
# from .prompt_encoder import PositionEmbeddingRandom
# from .twowaytransformer import TwoWayTransformer

# # From https://github.com/facebookresearch/segment-anything/blob/HEAD/segment_anything/modeling/mask_decoder.py
# class MaskDecoder(nn.Module):
#     def __init__(
#         self, 
#         args) -> None:
#     #     *,
#     #     transformer_dim: int,
#     #     transformer: nn.Module,
#     #     num_multimask_outputs: int = 3,
#     #     activation: Type[nn.Module] = nn.GELU,
#     #     iou_head_depth: int = 3,
#     #     iou_head_hidden_dim: int = 256,
#     # ) -> None:
#         """
#         Predicts masks given an image and prompt embeddings, using a
#         transformer architecture.

#         Arguments:
#           transformer_dim (int): the channel dimension of the transformer
#           transformer (nn.Module): the transformer used to predict masks
#           num_multimask_outputs (int): the number of masks to predict
#             when disambiguating masks
#           activation (nn.Module): the type of activation to use when
#             upscaling masks
#           iou_head_depth (int): the depth of the MLP used to predict
#             mask quality
#           iou_head_hidden_dim (int): the hidden dimension of the MLP
#             used to predict mask quality
#         """
#         super().__init__()
#         self.transformer_dim = args["dim_embed"]
#         self.transformer = TwoWayTransformer()

#         self.num_multimask_outputs = num_multimask_outputs

#         self.iou_token = nn.Embedding(1, transformer_dim)
#         self.num_mask_tokens = num_multimask_outputs + 1
#         self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

#         self.output_upscaling = nn.Sequential(
#             nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
#             LayerNorm2d(transformer_dim // 4),
#             activation(),
#             nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
#             activation(),
#         )
#         self.output_hypernetworks_mlps = nn.ModuleList(
#             [
#                 MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
#                 for i in range(self.num_mask_tokens)
#             ]
#         )

#         self.iou_prediction_head = MLP(
#             transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
#         )

#     def forward(
#         self,
#         image_embeddings: torch.Tensor,
#         image_pe: torch.Tensor,
#         sparse_prompt_embeddings: torch.Tensor,
#         dense_prompt_embeddings: torch.Tensor,
#         multimask_output: bool,
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Predict masks given image and prompt embeddings.

#         Arguments:
#           image_embeddings (torch.Tensor): the embeddings from the image encoder
#           image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
#           sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
#           dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
#           multimask_output (bool): Whether to return multiple masks or a single
#             mask.

#         Returns:
#           torch.Tensor: batched predicted masks
#           torch.Tensor: batched predictions of mask quality
#         """
#         masks, iou_pred = self.predict_masks(
#             image_embeddings=image_embeddings,
#             image_pe=image_pe,
#             sparse_prompt_embeddings=sparse_prompt_embeddings,
#             dense_prompt_embeddings=dense_prompt_embeddings,
#         )

#         # Select the correct mask or masks for output
#         if multimask_output:
#             mask_slice = slice(1, None)
#         else:
#             mask_slice = slice(0, 1)
#         masks = masks[:, mask_slice, :, :]
#         iou_pred = iou_pred[:, mask_slice]

#         # Prepare output
#         return masks, iou_pred

#     def predict_masks(
#         self,
#         image_embeddings: torch.Tensor,
#         image_pe: torch.Tensor,
#         sparse_prompt_embeddings: torch.Tensor,
#         dense_prompt_embeddings: torch.Tensor,
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """Predicts masks. See 'forward' for more details."""
#         # Concatenate output tokens
#         output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
#         output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
#         tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

#         # Expand per-image data in batch direction to be per-mask
#         src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
#         src = src + dense_prompt_embeddings
#         pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
#         b, c, h, w = src.shape

#         # Run the transformer
#         hs, src = self.transformer(src, pos_src, tokens)
#         iou_token_out = hs[:, 0, :]
#         mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

#         # Upscale mask embeddings and predict masks using the mask tokens
#         src = src.transpose(1, 2).view(b, c, h, w)
#         upscaled_embedding = self.output_upscaling(src)
#         hyper_in_list: List[torch.Tensor] = []
#         for i in range(self.num_mask_tokens):
#             hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
#         hyper_in = torch.stack(hyper_in_list, dim=1)
#         b, c, h, w = upscaled_embedding.shape
#         masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

#         # Generate mask quality predictions
#         iou_pred = self.iou_prediction_head(iou_token_out)

#         return masks, iou_pred