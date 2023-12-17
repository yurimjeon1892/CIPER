import torch
from torch import nn

from .common import MLP
from .transformer import TransformerDecoder, TransformerDecoderLayer


class Decoder(nn.Module):
    def __init__(self, args):
        """Initializes the model."""
        super().__init__()

        decoder_layer = TransformerDecoderLayer(
            d_model=args["dim_embed"],
            nhead=args["num_heads"],
            dim_feedforward=args["dim_feedforward"],
            dropout=args["dropout"],
            activation="relu",
            normalize_before=args["pre_norm"],
        )
        decoder_norm = nn.LayerNorm(args["dim_embed"])
        self.decoder = TransformerDecoder(
            decoder_layer,
            args["num_dec_layers"],
            decoder_norm,
            return_intermediate=False,
        )

        self.class_prediction_head = nn.Linear(args["dim_embed"], 2)
        self.bbox_prediction_head = MLP(
            args["dim_embed"], args["dim_embed"], output_dim=4, num_layers=3
        )

        self.num_queries = args["num_queries"]
        self.query_embed = nn.Embedding(self.num_queries, args["dim_embed"])

    def forward(self, x_grd, x_arl):
        """
        x_grd: bs x dim_embed
        x_arl: bs x num_patches x dim_embed
        """
        x_grd = torch.repeat_interleave(
            x_grd.unsqueeze(0), self.num_queries, dim=0
        )  # num_queries x bs x dim_embed
        query_pos = torch.repeat_interleave(
            self.query_embed.weight.unsqueeze(1), x_grd.shape[1], dim=1
        )  # num_queries x bs x dim_embed

        x_arl = x_arl.permute(1, 0, 2)  # num_patches x bs x dim_embed
        # print("input", x_grd.size(), query_pos.size(), x_arl.size(), pos.size())

        dst = self.decoder(tgt=x_grd, memory=x_arl, query_pos=query_pos)
        dst = dst.transpose(1, 2)  # 1 x bs x num_queries x dim_embed
        # print("dst: ", dst.size())

        class_pred = self.class_prediction_head(dst)  # 1 x bs x num_queries x 2
        bbox_pred = self.bbox_prediction_head(dst)  # 1 x bs x num_queries x 4
        # print("out: ", class_pred.size(), bbox_pred.size())

        out = {
            "pred_logits": class_pred[-1],
            "pred_boxes": bbox_pred[-1],
        }  # [-1]: last decoder layer output
        return out


# From https://github.com/facebookresearch/segment-anything/blob/HEAD/segment_anything/modeling/mask_decoder.py
from .prompt_encoder import PositionEmbeddingRandom
from .twowaytransformer import TwoWayTransformer


class TwoWayDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.iou_token = nn.Embedding(1, args["dim_embed"])
        self.mask_tokens = nn.Embedding(args["num_queries"], args["dim_embed"])

        self.pe_layer = PositionEmbeddingRandom(args["dim_embed"] // 2)

        self.transformer = TwoWayTransformer(
            depth=2,
            embedding_dim=args["dim_embed"],
            mlp_dim=2048,
            num_heads=8,
        )

        self.class_prediction_head = nn.Linear(args["dim_embed"], 2)
        self.bbox_prediction_head = MLP(
            args["dim_embed"], args["dim_embed"], output_dim=4, num_layers=3
        )

        # self.num_queries = args["num_queries"]
        self.image_embedding_size = (
            int(args["arl_img_size"][0] / args["patch_size"]),
            int(args["arl_img_size"][1] / args["patch_size"]),
        )

    def forward(
        self,
        prompt_embeddings: torch.Tensor,
        image_embeddings: torch.Tensor,
    ):
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          prompt_embeddings (torch.Tensor): the embeddings of the points and boxes [bs x dim_embed]
          image_embeddings (torch.Tensor): the embeddings from the image encoder [bs x num_patches x dim_embed]

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, prompt_embeddings.unsqueeze(1)), dim=1)

        # # Concatenate output tokens
        # output_tokens = self.iou_token.weight
        # output_tokens = output_tokens.unsqueeze(0).expand(
        #     prompt_embeddings.size(0), -1, -1
        # )
        # tokens = torch.cat((output_tokens, prompt_embeddings.unsqueeze(1)), dim=1)
        # tokens = prompt_embeddings.unsqueeze(1)

        src = image_embeddings

        image_pe = self.pe_layer(self.image_embedding_size).unsqueeze(
            0
        )  # 1 x dim_embed x h x w
        image_pe = image_pe.flatten(2).permute(0, 2, 1)  # 1 x num_patches x dim_embed
        pos_src = torch.repeat_interleave(
            image_pe, tokens.shape[0], dim=0
        )  # bs x num_patches x dim_embed

        # tokens = torch.repeat_interleave(tokens, self.num_queries, dim=1)

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs.unsqueeze(0)

        class_pred = self.class_prediction_head(
            iou_token_out
        )  # 1 x bs x num_queries x 2
        bbox_pred = self.bbox_prediction_head(iou_token_out)  # 1 x bs x num_queries x 4
        # print("out: ", class_pred.size(), bbox_pred.size())

        out = {
            "pred_logits": class_pred[-1],
            "pred_boxes": bbox_pred[-1],
        }  # [-1]: last decoder layer output
        return out
