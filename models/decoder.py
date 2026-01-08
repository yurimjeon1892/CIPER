import torch
from torch import nn

from .common import MLP
# From https://github.com/facebookresearch/segment-anything/blob/HEAD/segment_anything/modeling/mask_decoder.py
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
        self.num_mask_tokens = args["num_multimask_outputs"]
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, args["dim_embed"])

        self.iou_prediction_head = nn.Linear(args["dim_embed"], 1)
        self.bbox_prediction_head = nn.Linear(args["dim_embed"], 4)

    def forward(
        self,
        image_pe,
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

        # Concatenate output tokens
        # output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0) # (1 + num_queries) x dim_embed
        # output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1) # 1 x (1 + num_queries) x dim_embed
        # tokens = torch.cat((output_tokens, sparse_prompt_embeddings.unsqueeze(1)), dim=1) # 1 x (1 + num_queries + 1) x dim_embed
        ###
        sparse_prompt_embeddings = sparse_prompt_embeddings.unsqueeze(1)
        sparse_prompt_embeddings = torch.repeat_interleave(
            sparse_prompt_embeddings, self.num_mask_tokens, dim=1
        )
        tokens = sparse_prompt_embeddings

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(
            image_embeddings, tokens.shape[0], dim=0
        )  # 1 x num_patches x dim_embed
        # src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(
            image_pe.to(src.device), tokens.shape[0], dim=0
        )

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        # iou_token_out = hs[:, 0, :]
        # mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]
        mask_tokens_out = hs

        class_pred = self.iou_prediction_head(
            mask_tokens_out
        )  # 1 x (num_queries + 1) x 1
        bbox_pred = self.bbox_prediction_head(
            mask_tokens_out
        )  # 1 x (num_queries + 1) x 4

        return class_pred.squeeze(-1), bbox_pred
