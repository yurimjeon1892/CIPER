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
from common.utils import result2pose, target2gt


@torch.no_grad()
def inference_one(
    model: torch.nn.Module,
    postprocessors: torch.nn.Module,
    loader_dict: dict,
    eval_infos: dict,
):
    # retrieval validation
    model_query = model.query_net
    model_reference = model.reference_net

    model_query.eval()
    model_reference.eval()

    qry_label = np.zeros([len(loader_dict["qry"].dataset)])
    qry_feat = np.zeros([len(loader_dict["qry"].dataset), eval_infos["dim_feature"]])
    ref_feat = np.zeros([len(loader_dict["ref"].dataset), eval_infos["dim_feature"]])

    qry_fname_list = [None] * len(loader_dict["qry"].dataset)

    description = "[i] eval qry"
    for i, (img_grd, idx_grd, labels, file_name) in enumerate(
        tqdm(loader_dict["qry"], desc=description, unit="batches")
    ):

        img_grd = img_grd.to(eval_infos["device"])
        idx_grd = idx_grd.to(eval_infos["device"])
        labels = labels.to(eval_infos["device"])

        y_grd, _, _ = model_query(img_grd)
        qry_feat[idx_grd.cpu().numpy(), :] = y_grd.cpu().numpy()
        qry_label[idx_grd.cpu().numpy()] = labels.cpu().numpy()

        idx_grds = idx_grd.cpu().numpy()
        for j in range(idx_grds.shape[0]):
            qry_fname_list[idx_grds[j]] = file_name[j]

    ref_fname_list = [None] * len(loader_dict["ref"].dataset)

    description = "[i] eval ref"
    for i, (img_arl, idx_arl, _, file_name) in enumerate(
        tqdm(loader_dict["ref"], desc=description, unit="batches")
    ):
        img_arl = img_arl.to(eval_infos["device"])
        out_emb_arl, _, _ = model_reference(img_arl)  # delta

        ref_feat[idx_arl.cpu().numpy(), :] = out_emb_arl.detach().cpu().numpy()

        idx_arls = idx_arl.cpu().numpy()
        for j in range(idx_arls.shape[0]):
            ref_fname_list[idx_arls[j]] = file_name[j]

    N = qry_feat.shape[0]
    M = ref_feat.shape[0]

    qry_feat_norm = np.sqrt(np.sum(qry_feat**2, axis=1, keepdims=True))
    ref_feat_norm = np.sqrt(np.sum(ref_feat**2, axis=1, keepdims=True))
    similarity = np.matmul(
        qry_feat / qry_feat_norm, (ref_feat / ref_feat_norm).transpose()
    )

    os.makedirs("./infer", exist_ok=True)
    fname = "{data_name}-{eval_name}-retr.csv".format(
        data_name=eval_infos["data_name"],
        eval_name=eval_infos["eval_name"] + "_" + eval_infos["valid"],
    )

    f = open(os.path.join("./infer", fname), "w")
    f.write("qry, ref, top1, top2, top3, top4, top5\n")

    for i in range(N):
        f = open(os.path.join("./infer", fname), "a")
        data = ""
        data += qry_fname_list[i] + ","
        data += ref_fname_list[int(qry_label[i])] + ","

        indices = np.argsort(similarity[i, :])[::-1]

        data += ref_fname_list[indices[0]] + ","
        data += ref_fname_list[indices[1]] + ","
        data += ref_fname_list[indices[2]] + ","
        data += ref_fname_list[indices[3]] + ","
        data += ref_fname_list[indices[4]] + "\n"
        f.write(data)
        f.close()

    print("[i] evaluation finished. check: ", os.path.join("./infer", fname))
    return


@torch.no_grad()
def inference_two(
    model: torch.nn.Module,
    postprocessors: torch.nn.Module,
    loader_dict: dict,
    eval_infos: dict,
):
    os.makedirs("./infer", exist_ok=True)

    fname = "./{data_name}-{eval_name}-pose.csv".format(
        data_name=eval_infos["data_name"],
        eval_name=eval_infos["eval_name"] + "_" + eval_infos["valid"],
    )

    f = open(os.path.join("./infer", fname), "w")
    f.write(
        "filename, pred_lons, pred_lats, pred_oriens, gt_lons, gt_lats, gt_oriens\n"
    )

    model.eval()

    description = "[i] eval pose"
    for i, (img_grd, img_arl, targets) in enumerate(
        tqdm(loader_dict["val"], desc=description, unit="batches")
    ):
        f = open(os.path.join("./infer", fname), "a")
        img_grd = img_grd.to(eval_infos["device"])
        img_arl = img_arl.to(eval_infos["device"])
        targets_ = []
        for b in range(img_grd.size(0)):
            targets__ = {}
            for k in targets.keys():
                if k == "fname":
                    targets__[k] = targets[k][b]
                else:
                    targets__[k] = targets[k][b].to(eval_infos["device"])
            targets_.append(targets__)
        targets = targets_
        # targets = [
        #     {k: targets[k][b].to(eval_infos["device"]) for k in targets.keys()}
        #     for b in range(img_grd.size(0))
        # ]

        outputs = model(im_grd=img_grd, im_arl=img_arl)
        results = postprocessors(outputs, targets)

        for b in range(len(results)):

            pred_poses = np.squeeze(result2pose(results)[b])
            gt_poses = np.squeeze(target2gt(targets)[b])

            d = "{}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}\n".format(
                targets[b]["fname"],
                pred_poses[0],
                pred_poses[1],
                pred_poses[2],
                gt_poses[0],
                gt_poses[1],
                gt_poses[2],
            )
            f.write(d)

        f.close()

    print("[i] evaluation finished. check: ", os.path.join("./infer", fname))
    return


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

    inference_two(
        model,
        postprocessors,
        data_loader_valid_same,
        ({**eval_infos, **dict(valid="same")}),
    )
    if args["data_name"] == "kitti":
        inference_two(
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
