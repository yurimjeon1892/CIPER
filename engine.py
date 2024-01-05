"""
Train and eval functions used in main.py
"""
from typing import Iterable
from tqdm import tqdm

import torch
import numpy as np

import random
from common.utils import (
    AverageMeter,
    retr_accuracy,
    retr_accuracy_eval,
    pose_accuracy,
    pose_accuracy_eval,
)
from common.utils_plot import plot_result, plot_intermediate
import wandb
import datetime, os


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    postprocessors: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    train_infos: dict,
):
    model.train()
    # criterion.train()

    losses_meter = {}
    for k in criterion.losses:
        losses_meter[k] = AverageMeter()

    plot_imgs = {}

    iters = train_infos["iter"]

    sample_ind = random.choice(range(len(data_loader)))
    description = "[i] train {:>2}".format(train_infos["epoch"])
    for i, (img_grd, img_arl, targets) in enumerate(
        tqdm(data_loader, desc=description, unit="batches")
    ):
        bs = img_grd.size(0)
        img_grd = img_grd.to(train_infos["device"])
        img_arl = img_arl.to(train_infos["device"])

        outputs = model(im_grd=img_grd, im_arl=img_arl)

        targets = [
            {k: targets[k][b].to(train_infos["device"]) for k in targets.keys()}
            for b in range(bs)
        ]
        results = postprocessors(outputs, targets)
        if i == sample_ind:
            p_imgs = plot_result(results, targets, img_grd, img_arl)
            plot_imgs.update(p_imgs)

        loss_dict = criterion(outputs, targets)
        losses = sum(loss_dict[k] for k in loss_dict.keys())
        for k in loss_dict.keys():
            losses_meter[k].update(loss_dict[k].item(), bs)

        if i == sample_ind:
            p_imgs = plot_intermediate(criterion.intermediate)
            plot_imgs.update(p_imgs)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        losses.backward()
        # if train_infos["clip_max_norm"] > 0:
        #     torch.nn.utils.clip_grad_norm_(model.parameters(), train_infos["clip_max_norm"])
        if train_infos["optimizer"] != "sam":
            optimizer.step()
        else:
            optimizer.first_step(zero_grad=True)
            # second forward-backward pass, only for ASAM
            outputs = model(im_grd=img_grd, im_arl=img_arl)

            loss_dict = criterion(outputs, targets)
            losses = sum(loss_dict[k] for k in loss_dict.keys())
            losses.backward()
            optimizer.second_step(zero_grad=True)

        iters += bs
        # del loss_dict
        del outputs

    imgs = {}
    for k in plot_imgs.keys():
        if plot_imgs[k].shape[0] == 3:
            plot_imgs[k] = np.transpose(plot_imgs[k], (1, 2, 0))
        if wandb.run is not None:
            imgs["train_image/" + k] = wandb.Image(plot_imgs[k])
    if wandb.run is not None:
        wandb.log(imgs, step=train_infos["epoch"])

    stats = {}
    loss_total = 0
    for k in losses_meter.keys():
        stats["train_loss/" + k] = losses_meter[k].avg
        loss_total += losses_meter[k].avg
    stats["train_loss/total"] = loss_total
    if wandb.run is not None:
        wandb.log(stats, step=train_infos["epoch"])

    for k in stats.keys():
        print("   ", k + ": {:.8f}".format(stats[k]), end="\n")

    train_infos["iter"] = iters
    return train_infos


def valid_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    postprocessors: torch.nn.Module,
    loader_dict: dict,
    valid_infos: dict,
):
    imgs = dict()
    stats = dict()

    if True:
        # retrieval validation
        imgs1, stats1 = valid_retr(
            model, loader_dict["qry"], loader_dict["ref"], valid_infos
        )
        if valid_infos["valid"] == "same":
            valid_infos["metric"] = stats1["valid_same_acc/retr_top1"]
        if valid_infos["valid"] == "cross":
            valid_infos["metric"] = stats1["valid_cross_acc/retr_top1"]
        imgs.update(imgs1)
        stats.update(stats1)
        # wandb.log(img1)
        if wandb.run is not None:
            wandb.log(stats, step=valid_infos["epoch"])

    imgs2, stats2 = valid_pose(
        model, criterion, postprocessors, loader_dict["val"], valid_infos
    )
    imgs.update(imgs2)
    stats.update(stats2)
    if wandb.run is not None:
        wandb.log(imgs2, step=valid_infos["epoch"])
        wandb.log(stats2, step=valid_infos["epoch"])

    print("[i] Valid {:>2}:".format(valid_infos["epoch"]), end="\n")
    for k in stats.keys():
        print("   ", k + ": {:.8f}".format(stats[k]), end="\n")
    del valid_infos["valid"]
    return valid_infos


def valid_retr(
    model: torch.nn.Module,
    qry_loader: Iterable,
    ref_loader: Iterable,
    valid_infos: dict,
):
    model_query = model.query_net
    model_reference = model.reference_net

    model_query.eval()
    model_reference.eval()

    qry_label = np.zeros([len(qry_loader.dataset)])
    qry_feat = np.zeros([len(qry_loader.dataset), valid_infos["dim_feature"]])
    ref_feat = np.zeros([len(ref_loader.dataset), valid_infos["dim_feature"]])

    img_grd_, img_arl_ = None, None
    with torch.no_grad():
        # query features
        description = "[i] valid qry"
        for i, (img_grd, idx_grd, labels) in enumerate(
            tqdm(qry_loader, desc=description, unit="batches")
        ):
            img_grd = img_grd.to(valid_infos["device"])
            idx_grd = idx_grd.to(valid_infos["device"])
            labels = labels.to(valid_infos["device"])

            y_grd, _, _ = model_query(img_grd)
            qry_feat[idx_grd.cpu().numpy(), :] = y_grd.detach().cpu().numpy()
            qry_label[idx_grd.cpu().numpy()] = labels.detach().cpu().numpy()

            if i == 0:
                img_grd_ = img_grd[0, :, :, :]

        # reference features
        description = "[i] valid ref"
        for i, (img_arl, idx_arl, _) in enumerate(
            tqdm(ref_loader, desc=description, unit="batches")
        ):
            img_arl = img_arl.to(valid_infos["device"])
            out_emb_arl, _, _ = model_reference(img_arl)  # delta

            ref_feat[idx_arl.cpu().numpy(), :] = out_emb_arl.detach().cpu().numpy()
            if i == 0:
                img_arl_ = img_arl[0, :, :, :]

        retr_acc = retr_accuracy(qry_feat, ref_feat, qry_label.astype(int))

    imgs = {
        # "image/ir/grd": img_grd_,
        # "image/ir/arl": img_arl_,
    }
    if valid_infos["valid"] == "same":
        stats = {
            "valid_same_acc/retr_top1": retr_acc[0],
            "valid_same_acc/retr_top5": retr_acc[1],
            "valid_same_acc/retr_top10": retr_acc[2],
            "valid_same_acc/retr_top1pc": retr_acc[3],
        }
    elif valid_infos["valid"] == "cross":
        stats = {
            "valid_cross_acc/retr_top1": retr_acc[0],
            "valid_cross_acc/retr_top5": retr_acc[1],
            "valid_cross_acc/retr_top10": retr_acc[2],
            "valid_cross_acc/retr_top1pc": retr_acc[3],
        }
    else:
        raise (NotImplementedError())
    return imgs, stats


def valid_pose(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    postprocessors: torch.nn.Module,
    data_loader: Iterable,
    valid_infos: dict,
):
    model.eval()
    criterion.eval()

    losses_meter = {}
    for k in criterion.losses:
        losses_meter[k] = AverageMeter()

    trs_errs, rot_errs = [], []
    sample_ind = random.choice(range(len(data_loader)))
    # sample_ind = 0
    with torch.no_grad():
        description = "[i] valid pose"
        for i, (img_grd, img_arl, targets) in enumerate(
            tqdm(data_loader, desc=description, unit="batches")
        ):
            img_grd = img_grd.to(valid_infos["device"])
            img_arl = img_arl.to(valid_infos["device"])
            targets = [
                {k: targets[k][b].to(valid_infos["device"]) for k in targets.keys()}
                for b in range(img_grd.size(0))
            ]

            outputs = model(im_grd=img_grd, im_arl=img_arl)
            results = postprocessors(outputs, targets)

            loss_dict = criterion(outputs, targets)
            for k in loss_dict.keys():
                losses_meter[k].update(loss_dict[k].item(), img_grd.size(0))

            if i == sample_ind:
                plot_imgs = plot_result(results, targets, img_grd, img_arl)
                p_imgs = plot_intermediate(criterion.intermediate)
                plot_imgs.update(p_imgs)

            preds = result2pose(results)
            gts = target2gt(targets)

            trs_err, rot_err = pose_accuracy(preds, gts)
            trs_errs.extend(trs_err)
            rot_errs.extend(rot_err)

    imgs = {}
    stats = {}
    loss_total = 0
    if valid_infos["valid"] == "same":
        for k in plot_imgs.keys():
            if plot_imgs[k].shape[0] == 3:
                plot_imgs[k] = np.transpose(plot_imgs[k], (1, 2, 0))
            if wandb.run is not None:
                imgs["valid_same_image/" + k] = wandb.Image(plot_imgs[k])

        for k in losses_meter.keys():
            if "retr" in k:
                continue
            stats["valid_same_loss/" + k] = losses_meter[k].avg
            loss_total += losses_meter[k].avg

        stats["valid_same_loss/total"] = loss_total

        stats["valid_same_acc/pose_trs_d1"] = (
            np.sum((trs_err < 1)) / trs_err.shape[0] * 100
        )
        stats["valid_same_acc/pose_trs_d5"] = (
            np.sum((trs_err < 5)) / trs_err.shape[0] * 100
        )

        stats["valid_same_acc/pose_rot_d1"] = (
            np.sum((rot_err < 1)) / rot_err.shape[0] * 100
        )
        stats["valid_same_acc/pose_rot_d5"] = (
            np.sum((rot_err < 5)) / rot_err.shape[0] * 100
        )

        stats["valid_same_err/pose_trs_mean(m)"] = np.mean(trs_errs)
        stats["valid_same_err/pose_trs_median(m)"] = np.median(trs_errs)
        stats["valid_same_err/pose_rot_mean(deg)"] = np.mean(rot_errs)
        stats["valid_same_err/pose_rot_median(deg)"] = np.median(rot_errs)

    elif valid_infos["valid"] == "cross":
        for k in plot_imgs.keys():
            if plot_imgs[k].shape[0] == 3:
                plot_imgs[k] = np.transpose(plot_imgs[k], (1, 2, 0))
            if wandb.run is not None:
                imgs["valid_cross_image/" + k] = wandb.Image(plot_imgs[k])

        for k in losses_meter.keys():
            if "retr" in k:
                continue
            stats["valid_cross_loss/" + k] = losses_meter[k].avg
            loss_total += losses_meter[k].avg

        stats["valid_cross_loss/total"] = loss_total

        stats["valid_cross_acc/pose_trs_d1"] = (
            np.sum((trs_err < 1)) / trs_err.shape[0] * 100
        )
        stats["valid_cross_acc/pose_trs_d5"] = (
            np.sum((trs_err < 5)) / trs_err.shape[0] * 100
        )

        stats["valid_cross_acc/pose_rot_d1"] = (
            np.sum((rot_err < 1)) / rot_err.shape[0] * 100
        )
        stats["valid_cross_acc/pose_rot_d5"] = (
            np.sum((rot_err < 5)) / rot_err.shape[0] * 100
        )

        stats["valid_cross_err/pose_trs_mean(m)"] = np.mean(trs_errs)
        stats["valid_cross_err/pose_trs_median(m)"] = np.median(trs_errs)
        stats["valid_cross_err/pose_rot_mean(deg)"] = np.mean(rot_errs)
        stats["valid_cross_err/pose_rot_median(deg)"] = np.median(rot_errs)
    else:
        raise (NotImplementedError())

    return imgs, stats


@torch.no_grad()
def evaluate_one(
    model: torch.nn.Module,
    postprocessors: torch.nn.Module,
    loader_dict: dict,
    eval_infos: dict,
):
    os.makedirs("./eval", exist_ok=True)
    fname = "./eval/eval-{date:%Y-%m-%d-%H:%M:%S}.txt".format(
        date=datetime.datetime.now()
    )
    # retrieval validation
    model_query = model.query_net
    model_reference = model.reference_net

    model_query.eval()
    model_reference.eval()

    qry_label = np.zeros([len(loader_dict["qry"].dataset)])
    qry_feat = np.zeros([len(loader_dict["qry"].dataset), eval_infos["dim_feature"]])
    ref_feat = np.zeros([len(loader_dict["ref"].dataset), eval_infos["dim_feature"]])

    description = "[i] eval qry"
    for i, (img_grd, idx_grd, labels) in enumerate(
        tqdm(loader_dict["qry"], desc=description, unit="batches")
    ):
        img_grd = img_grd.to(eval_infos["device"])
        idx_grd = idx_grd.to(eval_infos["device"])
        labels = labels.to(eval_infos["device"])

        y_grd, _, _ = model_query(img_grd)
        qry_feat[idx_grd.cpu().numpy(), :] = y_grd.cpu().numpy()
        qry_label[idx_grd.cpu().numpy()] = labels.cpu().numpy()

    description = "[i] eval ref"
    for i, (img_arl, idx_arl, _) in enumerate(
        tqdm(loader_dict["ref"], desc=description, unit="batches")
    ):
        img_arl = img_arl.to(eval_infos["device"])
        out_emb_arl, _, _ = model_reference(img_arl)  # delta

        ref_feat[idx_arl.cpu().numpy(), :] = out_emb_arl.detach().cpu().numpy()

    retr_accuracy_eval(qry_feat, ref_feat, qry_label.astype(int), fname)

    model.eval()

    preds, gts = [], []
    description = "[i] eval pose"
    for i, (img_grd, img_arl, targets) in enumerate(
        tqdm(loader_dict["val"], desc=description, unit="batches")
    ):
        img_grd = img_grd.to(eval_infos["device"])
        img_arl = img_arl.to(eval_infos["device"])
        targets = [
            {k: targets[k][b].to(eval_infos["device"]) for k in targets.keys()}
            for b in range(img_grd.size(0))
        ]

        outputs = model(im_grd=img_grd, im_arl=img_arl)
        results = postprocessors(outputs, targets)

        preds.extend(result2pose(results))
        gts.extend(target2gt(targets))

    pose_accuracy_eval(preds, gts, fname)

    print("[i] evaluation finished. check: ", fname)
    return


def result2pose(results):
    pred_poses = []
    for b in range(len(results)):
        scores = results[b]["scores"].detach().cpu().numpy()
        shifts = results[b]["boxes"].detach().cpu().numpy()
        shifts_max = shifts[np.argmax(scores), :]
        shifts_max = np.array([[shifts_max[0], shifts_max[1], shifts_max[2]]])
        pred_poses.append(shifts_max)
    return pred_poses


def target2gt(targets):
    gts = []
    for b in range(len(targets)):
        arl_img_size = targets[b]["orig_size"].detach().cpu().numpy()
        meter_per_pixel = targets[b]["meter_per_pixel"][0].detach().cpu().numpy()

        tgt = targets[b]["boxes"][0].detach().cpu().numpy()
        tgt = np.array(
            [
                [
                    tgt[0] * arl_img_size[0] * meter_per_pixel,
                    tgt[1] * arl_img_size[1] * meter_per_pixel,
                    np.arctan2(tgt[3], tgt[2]),
                ]
            ]
        )
        gts.append(tgt)
    return gts
