import yaml
import argparse
import torch
import datetime
import sys, os

import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader


from datasets import build_dataset
from models import build

sys.path.append("../")

from common.utils_plot import *


def plot_infer_result(results, targets, img_grd, img_arl):

    img_grd_ = img_grd.detach().cpu().numpy()
    img_arl_ = img_arl.detach().cpu().numpy()

    for b in range(img_grd_.shape[0]):

        img_grd_b = img_grd_[b]
        img_arl_b = img_arl_[b]
        h, w = img_arl_b.shape[1], img_arl_b.shape[2]

        img_grd_b = (img_grd_b - np.min(img_grd_b)) / (
            np.max(img_grd_b) - np.min(img_grd_b)
        )
        img_arl_b = (img_arl_b - np.min(img_arl_b)) / (
            np.max(img_arl_b) - np.min(img_arl_b)
        )

        bev_mask = results[b]["bev_mask"].detach().cpu().numpy()

        n = int(bev_mask.shape[0] ** 0.5)
        bev_mask = np.reshape(bev_mask[:, 0], (n, n))
        img_bev_mask = draw_minmax_color_img(bev_mask, cmap=plt.cm.jet)
        img_bev_mask = Image.fromarray(img_bev_mask.astype(np.uint8))
        img_bev_mask = img_bev_mask.resize((h, w))

        img_arl_b_ = np.transpose(img_arl_b, (1, 2, 0)) * 255
        img_arl_b_ = Image.fromarray((img_arl_b_).astype(np.uint8))

        img_blend = Image.blend(img_arl_b_, img_bev_mask, alpha=0.4)

        arl_img_size = targets[b]["orig_size"].detach().cpu().numpy()
        meter_per_pixel = targets[b]["meter_per_pixel"][0].detach().cpu().numpy()

        tgt = targets[b]["boxes"][0].detach().cpu().numpy()
        yaw = np.arctan2(tgt[3], tgt[2])
        tgt = np.array(
            [
                [
                    tgt[0] * arl_img_size[0] * meter_per_pixel,
                    tgt[1] * arl_img_size[1] * meter_per_pixel,
                    yaw,
                ]
            ]
        )

        img_pin = np.array(img_blend)
        img_pin = draw_3dof_pin(img_pin, tgt, arl_img_size, meter_per_pixel, "orange")

        scores = results[b]["scores"].detach().cpu().numpy()
        shifts = results[b]["boxes"].detach().cpu().numpy()
        shifts_max = shifts[np.argmax(scores), :]
        shifts_max = np.array([[shifts_max[0], shifts_max[1], shifts_max[2]]])

        img_pin = draw_3dof_pin(
            img_pin, shifts_max, arl_img_size, meter_per_pixel, "cyan"
        )

        imgs = {
            str(b).zfill(2) + "_grd": img_grd_b * 255,
            # str(b).zfill(2) + "_ace": np.array(img_blend),
            str(b).zfill(2) + "_pin": img_pin,
        }

    return imgs


@torch.no_grad()
def inference_one(
    model: torch.nn.Module,
    postprocessors: torch.nn.Module,
    loader_dict: dict,
    eval_infos: dict,
):
    # # retrieval validation
    # model_query = model.query_net
    # model_reference = model.reference_net

    # model_query.eval()
    # model_reference.eval()

    # qry_label = np.zeros([len(loader_dict["qry"].dataset)])
    # qry_feat = np.zeros([len(loader_dict["qry"].dataset), eval_infos["dim_feature"]])
    # ref_feat = np.zeros([len(loader_dict["ref"].dataset), eval_infos["dim_feature"]])

    # if eval_infos["data_name"] == "vigor" or eval_infos["data_name"] == "kitti":
    #     description = "[i] eval qry"
    #     for i, (img_grd, idx_grd, labels) in enumerate(
    #         tqdm(loader_dict["qry"], desc=description, unit="batches")
    #     ):
    #         img_grd = img_grd.to(eval_infos["device"])
    #         idx_grd = idx_grd.to(eval_infos["device"])
    #         labels = labels.to(eval_infos["device"])

    #         y_grd, _, _ = model_query(img_grd)
    #         qry_feat[idx_grd.cpu().numpy(), :] = y_grd.cpu().numpy()
    #         qry_label[idx_grd.cpu().numpy()] = labels.cpu().numpy()

    #     description = "[i] eval ref"
    #     for i, (img_arl, idx_arl, _) in enumerate(
    #         tqdm(loader_dict["ref"], desc=description, unit="batches")
    #     ):
    #         img_arl = img_arl.to(eval_infos["device"])
    #         out_emb_arl, _, _ = model_reference(img_arl)  # delta

    #         ref_feat[idx_arl.cpu().numpy(), :] = out_emb_arl.detach().cpu().numpy()

    save_dir = "./infer/infer-{data_name}-{eval_name}-{date:%Y-%m-%d-%H:%M:%S}".format(
        data_name=eval_infos["data_name"],
        eval_name=eval_infos["eval_name"] + "_" + eval_infos["valid"],
        date=datetime.datetime.now(),
    )
    os.makedirs(save_dir, exist_ok=True)
    f = open(os.path.join(save_dir, "pose.txt"), "w")

    model.eval()

    description = "[i] eval pose"
    for i, (img_grd, img_arl, targets) in enumerate(
        tqdm(loader_dict["val"], desc=description, unit="batches")
    ):
        if i % 10 != 0:
            continue
        img_grd = img_grd.to(eval_infos["device"])
        img_arl = img_arl.to(eval_infos["device"])
        targets = [
            {k: targets[k][b].to(eval_infos["device"]) for k in targets.keys()}
            for b in range(img_grd.size(0))
        ]

        outputs = model(im_grd=img_grd, im_arl=img_arl)
        results = postprocessors(outputs, targets)

        for b in range(len(results)):

            pred_poses = np.squeeze(result2pose(results)[b])
            gt_poses = np.squeeze(target2gt(targets)[b])

            d = "{:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}\n".format(
                gt_poses[0],
                gt_poses[1],
                gt_poses[2],
                pred_poses[0],
                pred_poses[1],
                pred_poses[2],
            )
            f.write(d)

        plot_imgs = plot_infer_result(results, targets, img_grd, img_arl)
        for k in plot_imgs.keys():

            fn = os.path.join(save_dir, str(i).zfill(6) + "_" + k + ".png")
            if plot_imgs[k].shape[-1] != 3:
                im = np.transpose(plot_imgs[k], (1, 2, 0))
            else:
                im = plot_imgs[k]
            im = Image.fromarray(im.astype(np.uint8))
            im.save(fn)

    f.close()

    print("[i] evaluation finished. check: ", save_dir)
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


def inference():
    ## init model
    model, _, postprocessors = build(args)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("[i] number of params:", n_parameters // 10**6, "M")

    ## load pretrained
    checkpoint = torch.load(args["pretrained"], map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    print("[i] load checkpoint from:", args["pretrained"], "for evaluation")

    ## set data loader for image retrieval validation
    dataset_val_s_q = build_dataset(mode="valid_same_qry", args=args)
    dataset_val_s_r = build_dataset(mode="valid_same_ref", args=args)

    data_loader_val_s_q = DataLoader(
        dataset_val_s_q,
        batch_size=32,
        shuffle=False,
        num_workers=args["num_workers"],
        pin_memory=True,
    )
    data_loader_val_s_r = DataLoader(
        dataset_val_s_r,
        batch_size=64,
        shuffle=False,
        num_workers=args["num_workers"],
        pin_memory=True,
    )

    data_loader_valid_same = {
        "qry": data_loader_val_s_q,
        "ref": data_loader_val_s_r,
    }

    if args["data_name"] == "kitti":
        dataset_val_c_q = build_dataset(mode="valid_cross_qry", args=args)
        dataset_val_c_r = build_dataset(mode="valid_cross_ref", args=args)
        data_loader_val_c_q = DataLoader(
            dataset_val_c_q,
            batch_size=32,
            shuffle=False,
            num_workers=args["num_workers"],
            pin_memory=True,
        )
        data_loader_val_c_r = DataLoader(
            dataset_val_c_r,
            batch_size=64,
            shuffle=False,
            num_workers=args["num_workers"],
            pin_memory=True,
        )

        data_loader_valid_cross = {
            "qry": data_loader_val_c_q,
            "ref": data_loader_val_c_r,
        }

    ## set data loader for pose estimation validation
    dataset_val_same = build_dataset(mode="valid_same", args=args)
    data_loader_val_same = DataLoader(
        dataset_val_same,
        batch_size=args["batch_size"],
        shuffle=False,
        drop_last=False,
        num_workers=args["num_workers"],
    )
    data_loader_valid_same["val"] = data_loader_val_same

    if args["data_name"] == "kitti":
        dataset_val_cross = build_dataset(mode="valid_cross", args=args)
        data_loader_val_cross = DataLoader(
            dataset_val_cross,
            batch_size=args["batch_size"],
            shuffle=False,
            drop_last=False,
            num_workers=args["num_workers"],
        )
        data_loader_valid_cross["val"] = data_loader_val_cross

    ## set infos for evaluation
    print("[i] start inference ~")

    eval_infos = {
        "device": args["device"],
        "dim_feature": args["dim_feature"],
        "data_name": args["data_name"],
        "eval_name": args["eval_name"],
    }
    inference_one(
        model,
        postprocessors,
        data_loader_valid_same,
        ({**eval_infos, **dict(valid="same")}),
    )
    if args["data_name"] == "kitti":
        inference_one(
            model,
            postprocessors,
            data_loader_valid_cross,
            ({**eval_infos, **dict(valid="cross")}),
        )
    return


def parse_args():
    parser = argparse.ArgumentParser(description="Train a CIPER")
    parser.add_argument("config", help="config file path")
    parser.add_argument(
        "--debug", action="store_true", help="debug flag for disble logger"
    )
    args = parser.parse_args()
    return args


def main():
    cmd_args = parse_args()

    global args
    with open(cmd_args.config, "r") as stream:
        args = yaml.safe_load(stream)

    if args["eval"]:
        inference()
    else:
        print("[!] wrong option")

    return


if __name__ == "__main__":
    main()
