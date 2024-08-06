# From https://github.com/Jeff-Zilence/TransGeo2022/blob/main/model/Deit.py

import torch
from torch import nn

from functools import partial
from timm.models.vision_transformer import VisionTransformer

import torchvision
import numpy as np


class Encoder(VisionTransformer):
	def __init__(
		self,
		args,
		img_size,
		norm_layer=partial(nn.LayerNorm, eps=1e-6),
	):
		super().__init__(
			img_size=img_size,
			patch_size=args["patch_size"],
			embed_dim=args["dim_embed"],
			num_classes=args["dim_feature"],
			depth=args["num_enc_layers"],
			num_heads=args["num_heads"],
			mlp_ratio=args["mlp_ratio"],
			qkv_bias=args["qkv_bias"],
			norm_layer=norm_layer,
		)

		num_patches = self.patch_embed.num_patches

		self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 3, self.embed_dim))
		self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
		self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
		self.third_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

		nn.init.trunc_normal_(self.cls_token)
		nn.init.trunc_normal_(self.dist_token)
		nn.init.trunc_normal_(self.third_token)

		self.head = nn.Linear(self.embed_dim, self.num_classes)
		self.head_dist = nn.Linear(self.embed_dim, self.num_classes)
		self.head.apply(self._init_weights)
		self.head_dist.apply(self._init_weights)

		# self.prompt_pe = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))
		self._load_pretrained(img_size, args["dim_feature"], args["patch_size"])		
		
	def get_dense_pe(self):
		return nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, self.embed_dim))
			

	def _load_pretrained(self, img_size, num_classes, patch_size):
		checkpoint = torch.hub.load_state_dict_from_url(
			"https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth",
			map_location="cpu",
		)

		weight = checkpoint["model"]["pos_embed"]
		ori_size = np.sqrt(weight.shape[1] - 1).astype(int)
		new_size = (
			img_size[0] // self.patch_embed.patch_size[0],
			img_size[1] // self.patch_embed.patch_size[1],
		)
		matrix = (
			weight[:, 2:, :]
			.reshape([1, ori_size, ori_size, weight.shape[-1]])
			.permute((0, 3, 1, 2))
		)
		resize = torchvision.transforms.Resize(new_size)
		new_matrix = (
			resize(matrix).permute(0, 2, 3, 1).reshape([1, -1, weight.shape[-1]])
		)
		checkpoint["model"]["pos_embed"] = torch.cat(
			[weight[:, :2, :], weight[:, :1, :], new_matrix], dim=1
		)
		# checkpoint["model"]["prompt_pe"] = new_matrix
		checkpoint["model"]["third_token"] = weight[:, :1, :]
		# change the prediction head if not 1000
		if num_classes != 1000:
			checkpoint["model"]["head.weight"] = checkpoint["model"][
				"head.weight"
			].repeat(5, 1)[:num_classes, :]
			checkpoint["model"]["head.bias"] = checkpoint["model"]["head.bias"].repeat(
				5
			)[:num_classes]
			checkpoint["model"]["head_dist.weight"] = checkpoint["model"][
				"head.weight"
			].repeat(5, 1)[:num_classes, :]
			checkpoint["model"]["head_dist.bias"] = checkpoint["model"][
				"head.bias"
			].repeat(5)[:num_classes]
		if patch_size != 16:
			checkpoint["model"]["patch_embed.proj.weight"] = checkpoint["model"][
				"patch_embed.proj.weight"
			].repeat(1, 1, int(patch_size / 16), int(patch_size / 16))
		msg = self.load_state_dict(checkpoint["model"])
		print(msg)

	def forward_features(self, x):
		# taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
		# with slight modifications to add the dist_token
		B = x.shape[0]
		x = self.patch_embed(x)

		cls_tokens = self.cls_token.expand(
			B, -1, -1
		)  # stole cls_tokens impl from Phil Wang, thanks
		dist_tokens = self.dist_token.expand(B, -1, -1)
		third_tokens = self.third_token.expand(B, -1, -1)
		x = torch.cat((cls_tokens, dist_tokens, third_tokens, x), dim=1)

		x = x + self.pos_embed
		x = self.pos_drop(x)

		for i, blk in enumerate(self.blocks):
			x = blk(x)

		x = self.norm(x)
		return (x[:, 0], x[:, 1]), x[:, 2], x[:, 3:]

	def forward(self, x):
		x1, x2, x3 = self.forward_features(x)
		x1_1 = self.head(x1[0])
		x1_2 = self.head_dist(x1[1])
		return (x1_1 + x1_2) / 2, x2, x3
