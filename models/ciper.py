# From https://github.com/facebookresearch/detr/blob/HEAD/models/detr.py

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn.functional as F
from torch import nn
import math

from .matcher import build_matcher
from .soft_triplet import SoftTripletBiLoss

from .encoder import Encoder
from .recoder import Recoder
from .decoder import Decoder, TwoWayDecoder


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

        self.query_net = Encoder(args, args["grd_img_size"], mode="query")
        self.reference_net = Encoder(args, args["arl_img_size"])
        self.retr_only = args["retr_only"]
        self.rng_mask = args["rng_mask"]
        if not self.retr_only:
            self.rot_net = Recoder(args)
            self.pose_net = TwoWayDecoder(args)

    def forward(self, im_grd, im_arl):
        y1_grd, y2_grd, y3_grd = self.query_net(im_grd)
        y1_arl, _, y3_arl = self.reference_net(im_arl)
        outputs = {
            "grd": y1_grd,
            "arl": y1_arl,
        }
        if not self.retr_only:
            if self.rng_mask:
                masks = self.rot_net(y3_grd, y3_arl)
                y3_arl = torch.mul(masks["bev_mask"].to(y3_arl.device), y3_arl)
                outputs.update(masks)
            out_pos = self.pose_net(y2_grd, y3_arl)
            outputs.update(out_pos)

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
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        self.intermediate = {}

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
        src_logits = outputs["pred_logits"]  # bs x num_queries x 2

        # idx = self._get_src_permutation_idx(indices)
        # target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        # target_classes = torch.full(src_logits.shape[:2], self.num_classes,
        #                             dtype=torch.int64, device=src_logits.device)
        # target_classes[idx] = target_classes_o

        # loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        # losses = {"labels": loss_ce}

        idx = self._get_src_permutation_idx(indices)
        target_classes = torch.zeros_like(src_logits)
        target_classes[:, :, -1] = 1.0
        target_classes[idx] = torch.tensor([1.0, 0.0]).to(src_logits.device)

        # print("aa", torch.sum(target_classes[:, :, 0]), torch.sum(target_classes[:, :, 1]), torch.sum(target_classes, dim=2))

        # o = torch.flatten(src_logits, 0, 1)
        # t = torch.flatten(target_classes, 0, 1)
        o, t = src_logits, target_classes
        loss_bce = F.binary_cross_entropy_with_logits(o, t.float(), self.empty_weight)
        losses = {"labels": loss_bce}

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

        # loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        loss_bbox = F.mse_loss(src_boxes, target_boxes.float(), reduction="none")

        losses = {}
        losses["boxes"] = loss_bbox.sum()

        return losses

    def loss_mask(self, outputs, targets, indices):
        src_mask = outputs["rng_mask"]
        bs, w = src_mask.size(0), src_mask.size(-1)
        marg_w = int(w / 8)

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

        loss_mask = F.binary_cross_entropy(src_mask, target_mask)

        losses = {}
        losses["mask"] = loss_mask

        self.intermediate["target_mask"] = target_mask
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
        if "labels" in self.losses:
            indices = self.matcher(outputs, targets)
        else:
            indices = None

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

        # prob = F.softmax(out_logits, -1)
        # scores, labels = prob[..., :-1].max(-1)

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

        rng_mask, bev_mask = outputs["rng_mask"], outputs["bev_mask"]

        # results = [{"scores": s, "labels": l, "boxes": b} for s, l, b in zip(scores, labels, boxes)]
        # results = [{"scores": s, "boxes": b} for s, b in zip(scores, boxes)]
        results = [
            {"scores": s, "boxes": b, "rng_mask": rm, "bev_mask": bm}
            for s, b, rm, bm in zip(scores, boxes, rng_mask, bev_mask)
        ]
        return results


def build(args):
    model = CIPER(args)

    # build criterion
    if args["retr_only"]:
        matcher = None
        weight_dict = {"retrieval": 1}
        eos_coef = 0
        losses = ["retrieval"]
    else:
        matcher = build_matcher(args)
        weight_dict = {
            "retrieval": 1,
            "labels": args["label_loss_coef"],
            "boxes": args["bbox_loss_coef"],
            "mask": 5.0,
        }
        eos_coef = args["eos_coef"]
        losses = ["retrieval", "labels", "boxes", "mask"]
    criterion = SetCriterion(
        matcher=matcher, weight_dict=weight_dict, eos_coef=eos_coef, losses=losses
    )

    # build post processor
    if args["retr_only"]:
        postprocessors = None
    else:
        postprocessors = PostProcess()

    return model.to(args["device"]), criterion.to(args["device"]), postprocessors
