import torch
from torch import nn
from .common import MLP

# From https://github.com/facebookresearch/segment-anything/blob/HEAD/segment_anything/modeling/mask_decoder.py
from .prompt_encoder import PositionEmbeddingRandom
from .twowaytransformer import TwoWayTransformer


class TwoWayDecoder(nn.Module):
	def __init__(self, args):
		super().__init__()

		self.transformer = TwoWayTransformer(
			depth=2,
			embedding_dim=args["dim_embed"],
			mlp_dim=2048,
			num_heads=8,
		)

		self.iou_token = nn.Embedding(1, args["dim_embed"])
		self.num_mask_tokens = args["num_multimask_outputs"] + 1
		self.mask_tokens = nn.Embedding(self.num_mask_tokens, args["dim_embed"])

		self.pe_layer = PositionEmbeddingRandom(args["dim_embed"] // 2)		
		
		self.iou_prediction_head = MLP(
            args["dim_embed"], 256, output_dim=self.num_mask_tokens, num_layers=3
        )
		self.bbox_prediction_head = MLP(
			args["dim_embed"], args["dim_embed"], output_dim=4, num_layers=3
		)

		self.image_embedding_size = (
			int(args["arl_img_size"][0] / args["patch_size"]),
			int(args["arl_img_size"][1] / args["patch_size"]),
		)

	def forward(
		self,
		image_embeddings: torch.Tensor,
		sparse_prompt_embeddings: torch.Tensor,
	):
		"""
		Predict masks given image and prompt embeddings.

		Arguments:
		  image_embeddings (torch.Tensor): the embeddings from the image encoder [1 x num_patches x dim_embed]
		  sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes [1 x dim_embed]

		Returns:
		  torch.Tensor: batched predicted masks
		  torch.Tensor: batched predictions of mask quality
		"""
		# Make image pe
		image_pe = self.pe_layer(self.image_embedding_size).unsqueeze(0)  # 1 x dim_embed x h x w
		image_pe = image_pe.flatten(2).permute(0, 2, 1)  # 1 x num_patches x dim_embed
		# print("image_pe", image_pe.size())

		# Concatenate output tokens
		output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0) # (1 + num_queries) x dim_embed
		output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1) # 1 x (1 + num_queries) x dim_embed
		tokens = torch.cat((output_tokens, sparse_prompt_embeddings.unsqueeze(1)), dim=1) # 1 x (1 + num_queries + 1) x dim_embed
		# print("tokens", tokens.size())

		# Expand per-image data in batch direction to be per-mask
		# src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0) # 1 x num_patches x dim_embed
		src = image_embeddings
		# src = src + dense_prompt_embeddings
		pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
		# print("pos_src", pos_src.size(), src.size())
		# b, c, hw = src.shape

		# Run the transformer
		hs, src = self.transformer(src, pos_src, tokens)
		iou_token_out = hs[:, 0, :]
		mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

		class_pred = self.iou_prediction_head(iou_token_out)  # 1 x (num_queries + 1)
		bbox_pred = self.bbox_prediction_head(mask_tokens_out)  # 1 x (num_queries + 1) x 4
		# print(class_pred.size(), bbox_pred.size());exit()

		return class_pred, bbox_pred
