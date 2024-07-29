# From https://github.com/facebookresearch/detr/blob/HEAD/models/detr.py

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn.functional as F
from torch import nn
import math

from .matcher import build_matcher
from .soft_triplet import SoftTripletBiLoss

from .encoder import Encoder
from .ace import AeroConfidenceEstimator
from .decoder import TwoWayDecoder


class CIPER(nn.Module):
	"""This is the CIPER module that performs cross-view image geo-localization"""

	def __init__(self, args):
		"""Initializes the model.
		Parameters:
			backbone: torch module of the backbone to be used. See backbone.py
			num_queries: number of object queries, ie detection slot. This is the maximal number of objects
						 DETR can detect in a single image. For COCO, we recommend 100 queries.
		"""
		super().__init__()

		self.query_net = Encoder(args, args["grd_img_size"])
		self.reference_net = Encoder(args, args["arl_img_size"])
		self.two_way_decoder = TwoWayDecoder(args)
		self.mask = args["mask"]
		if self.mask: self.ace_mask_net = AeroConfidenceEstimator(args)
		self.batch_size = args["batch_size"]

	def forward(self, im_grd, im_arl):
		x1_grd, x2_grd, x3_grd = self.query_net(im_grd)
		x1_arl, _, x3_arl = self.reference_net(im_arl)
		outputs = {
			"grd": x1_grd,
			"arl": x1_arl,
		}
		if self.mask:
			masks = self.ace_mask_net(x3_grd, x3_arl)			
			x3_arl = torch.mul(masks["bev_mask"], x3_arl)
			outputs.update(masks)

		out_pred_logits, out_pred_boxes = [], []
		for b in range(x2_grd.size(0)):
			pred_logits, pred_boxes = self.two_way_decoder(
				sparse_prompt_embeddings=x2_grd[b].unsqueeze(0),
				image_embeddings=x3_arl[b].unsqueeze(0))		
			# print(pred_logits.size(), pred_boxes.size())
			out_pred_logits.append(pred_logits)
			out_pred_boxes.append(pred_boxes)
		
		# exit()
		outputs["pred_logits"] = torch.cat(out_pred_logits, 0)
		outputs["pred_boxes"] = torch.cat(out_pred_boxes, 0)
		return outputs


class SetCriterion(nn.Module):
	def __init__(self, matcher, weight_dict, eos_coef, losses):
		"""Create the criterion.
		Parameters:
		"""
		super().__init__()

		self.num_classes = 1

		self.matcher = matcher
		self.weight_dict = weight_dict
		self.eos_coef = eos_coef
		self.losses = losses
		# empty_weight = torch.ones(self.num_classes + 1)
		# empty_weight[-1] = self.eos_coef
		# self.register_buffer("empty_weight", empty_weight)

		if "retrieval" in losses:
			self.soft_triplet_loss = SoftTripletBiLoss().cuda()

	def loss_retrieval(self, outputs, targets, indices):
		loss_ir, mean_p, mean_n = self.soft_triplet_loss(outputs["grd"], outputs["arl"])
		losses = {"retrieval": loss_ir}
		return losses

	def loss_labels(self, outputs, targets, indices):
		"""Classification loss (NLL)
		targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
		"""
		assert "pred_logits" in outputs
		src_logits = outputs["pred_logits"].sigmoid()  # bs x num_queries

		idx = self._get_src_permutation_idx(indices)
		# target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
		target_classes = torch.full(src_logits.shape[:2], 0, dtype=torch.int64, device=src_logits.device)
		target_classes[idx] = 1

		loss_ce = F.binary_cross_entropy(src_logits, target_classes.float())
		losses = {"labels": loss_ce}

		return losses

	def loss_boxes(self, outputs, targets, indices):
		"""Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
		targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
		The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
		"""
		assert "pred_boxes" in outputs
		idx = self._get_src_permutation_idx(indices)
		src_boxes = outputs["pred_boxes"][idx]
		target_boxes = torch.cat(
			[t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
		)

		loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

		losses = {}
		losses["boxes"] = loss_bbox.sum()

		return losses

	def loss_mask(self, outputs, targets, indices):
		# this is just for debug
		src_mask = outputs["rng_mask"]
		bs, w = src_mask.size(0), src_mask.size(-1)
		marg_w = int(w / 5)

		target_boxes = torch.cat(
			[t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
		)
		c, s = target_boxes[:, 2], target_boxes[:, 3]
		yaw = torch.atan2(s, c)
		yaw[yaw < 0] = yaw[yaw < 0] + 2 * math.pi

		rounded_tensor = (yaw / (2 * math.pi)) * w
		rounded_tensor = torch.clamp(rounded_tensor.round(), 0, w - 1).long()

		target_mask = torch.zeros(src_mask.size(), requires_grad=False)
		for i in range(bs):
			y_id = rounded_tensor[i]
			if y_id < marg_w:
				target_mask[i, :, : y_id + marg_w] = 1.0
				target_mask[i, :, y_id - marg_w :] = 1.0
			elif y_id > w - marg_w:
				target_mask[i, :, : w - y_id] = 1.0
				target_mask[i, :, y_id - marg_w :] = 1.0
			else:
				target_mask[i, :, y_id - marg_w : y_id + marg_w] = 1.0

		# real loss calc
		src_cos_sin = outputs["pred_cos_sin"]
		target_boxes = torch.cat(
			[t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
		)
		loss_cos_sin = F.mse_loss(src_cos_sin.float(), target_boxes[:, 2:].float())

		losses = {}
		losses["mask"] = loss_cos_sin

		return losses

	def _get_src_permutation_idx(self, indices):
		# permute predictions following indices
		batch_idx = torch.cat(
			[torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
		)
		src_idx = torch.cat([src for (src, _) in indices])
		return batch_idx, src_idx

	def get_loss(self, loss, outputs, targets, indices):
		loss_map = {
			"retrieval": self.loss_retrieval,
			"labels": self.loss_labels,
			"boxes": self.loss_boxes,
			"mask": self.loss_mask,
		}
		assert loss in loss_map, f"do you really want to compute {loss} loss?"
		return loss_map[loss](outputs, targets, indices)

	def forward(self, outputs, targets):
		"""This performs the loss computation.
		Parameters:
			 outputs: dict of tensors, see the output specification of the model for the format
			 targets: list of dicts, such that len(targets) == batch_size.
					  The expected keys in each dict depends on the losses applied, see each loss" doc
		"""
		indices = self.matcher(outputs, targets)

		# Compute all the requested losses
		losses = {}
		for loss in self.losses:
			losses.update(self.get_loss(loss, outputs, targets, indices))

		for k in losses.keys():
			losses[k] = losses[k] * self.weight_dict[k]

		return losses


class PostProcess(nn.Module):
	"""This module converts the model's output into the format expected by the coco api"""

	@torch.no_grad()
	def forward(self, outputs, targets):
		"""Perform the computation
		Parameters:
			outputs: raw outputs of the model
			target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
						  For evaluation, this must be the original image size (before any data augmentation)
						  For visualization, this should be the image size after data augment, but before padding
		"""
		out_logits, out_bbox = (
			outputs["pred_logits"],
			outputs["pred_boxes"],
		)  # bs x num_quries x 4

		assert len(out_logits) == len(targets)

		prob = torch.sigmoid(out_logits)
		scores = prob[..., :-1]

		x_c, y_c, c, s = out_bbox.unbind(-1)  # bs x num_quries
		yaw = torch.atan2(s, c)

		xs, ys = [], []
		for b in range(len(out_logits)):
			arl_img_size = targets[b]["orig_size"]
			meter_per_pixel = targets[b]["meter_per_pixel"][0]
			x = x_c[b] * arl_img_size[0] * meter_per_pixel
			y = y_c[b] * arl_img_size[1] * meter_per_pixel
			xs.append(x)
			ys.append(y)
		xs = torch.stack(xs, 0)
		ys = torch.stack(ys, 0)

		boxes = torch.stack([xs, ys, yaw], dim=-1)

		results = [{"scores": s, "boxes": b} for s, b in zip(scores, boxes)]
		# rng_mask, bev_mask = outputs["rng_mask"], outputs["bev_mask"]
		# results = [
		#     {"scores": s, "boxes": b, "rng_mask": rm, "bev_mask": bm}
		#     for s, b, rm, bm in zip(scores, boxes, rng_mask, bev_mask)
		# ]
		return results


def build(args):
	model = CIPER(args)
	model = model.to(args["device"])

	if args["eval"]:
		criterion = None
	else:
		# build criterion
		matcher = build_matcher(args)
		weight_dict = {
			"retrieval": args["retrieval_loss_coef"],
			"labels": args["label_loss_coef"],
			"boxes": args["bbox_loss_coef"],
			"mask": args["mask_loss_coef"],
		}
		eos_coef = args["eos_coef"]
		if args["mask"]:
			losses = ["retrieval", "labels", "boxes", "mask"]
		else:
			losses = ["retrieval", "labels", "boxes"]
		criterion = SetCriterion(
			matcher=matcher, weight_dict=weight_dict, eos_coef=eos_coef, losses=losses
		)
		criterion = criterion.to(args["device"])

	# build post processor
	postprocessors = PostProcess()

	return model, criterion, postprocessors
