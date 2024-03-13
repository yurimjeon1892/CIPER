import yaml
import argparse
import torch
import sys, os

import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader
import datetime

from datasets import build_dataset
from models import build

from prompt_toolkit import prompt
from prompt_toolkit.completion import PathCompleter

sys.path.append("../")

from common.utils import result2pose
from common.utils_inference import *


@torch.no_grad()
def stage_one(
    model: torch.nn.Module,
    grd_qry: torch.Tensor,
    db_loader: torch.utils.data.DataLoader,
    infos: dict,
):
    # prepare model
    model_query = model.query_net
    model_reference = model.reference_net

    model_query.eval()
    model_reference.eval()

    # run query_net
    description = "[i] image retrieval - qry"
    print(description)
    qry_feat = np.zeros([1, infos["dim_feature"]])
    grd_qry = grd_qry.to(infos["device"])
    y_grd, _, _ = model_query(grd_qry)
    qry_feat[0, :] = y_grd.cpu().numpy()

    # run reference_net
    description = "[i] image retrieval - data_base"

    meta_info_list = [None] * len(db_loader.dataset)

    ref_feat = np.zeros([len(db_loader.dataset), infos["dim_feature"]])
    for i, (img_arl, idx_arl, meta_info) in enumerate(
        tqdm(db_loader, desc=description, unit="batches")
    ):
        img_arl = img_arl.to(infos["device"])
        out_emb_arl, _, _ = model_reference(img_arl)  # delta

        ref_feat[idx_arl.cpu().numpy(), :] = out_emb_arl.detach().cpu().numpy()

        idx_arls = idx_arl.cpu().numpy()
        for j in range(idx_arls.shape[0]):
            tmp_dict = {}
            for k in meta_info.keys():
                tmp_dict[k] = meta_info[k][j]
            meta_info_list[idx_arls[j]] = tmp_dict

    # compute similarity
    qry_feat_norm = np.sqrt(np.sum(qry_feat**2, axis=1, keepdims=True))
    ref_feat_norm = np.sqrt(np.sum(ref_feat**2, axis=1, keepdims=True))
    similarity = np.matmul(
        qry_feat / qry_feat_norm, (ref_feat / ref_feat_norm).transpose()
    )

    # sort similarity
    indices = list(np.argsort(similarity[0, :])[::-1])

    meta_info_top_k = []
    for ind in indices:
        meta_info_top_k.append(meta_info_list[ind])

    print("[i] image retrieval finished")
    return meta_info_top_k[: infos["top_k"]]


@torch.no_grad()
def stage_two(
    model: torch.nn.Module,
    postprocessors: torch.nn.Module,
    grd_qry: torch.Tensor,
    arl_candidates: list,
    infos: dict,
):

    model.eval()

    description = "[i] pose estimation"
    print(description, len(arl_candidates))

    pred_poses = []

    for i, arl_img in enumerate(arl_candidates):

        img_grd = grd_qry.to(infos["device"])
        img_arl = arl_img.to(infos["device"])

        outputs = model(im_grd=img_grd, im_arl=img_arl)
        results = postprocessors(outputs)

        pred_pose = result2pose(results)
        pred_poses.append(pred_pose)

    print("[i] pose estimation finished")
    return pred_poses


def inference():
    ## init model
    model, _, _ = build(args)

    ## init post processor
    postprocessors = PostProcess()

    ## load pretrained
    checkpoint = torch.load(args["pretrained"], map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    print("[i] load checkpoint from:", args["pretrained"], "for evaluation")

    ## set infos
    infos = {
        "device": args["device"],
        "dim_feature": args["dim_feature"],
        "top_k": args["top_k"],
    }

    ## set data loader for image retrieval
    dataset_db = build_dataset("infer", args)
    data_loader_db = DataLoader(
        dataset_db,
        batch_size=64,
        shuffle=False,
        num_workers=args["num_workers"],
        pin_memory=True,
    )

    ## set infos for evaluation
    print("[i] start inference ~")

    ## set query
    grd_qry = load_query(args)
    # image retrieval
    pred_meta_topk = stage_one(model, grd_qry, data_loader_db, infos)
    # load candidates
    arl_candidates = load_candidates(args, pred_meta_topk)
    # pose estimation
    pred_pose = stage_two(
        model,
        postprocessors,
        grd_qry,
        arl_candidates,
        infos,
    )
    # combine results
    pred_locs = run_geo_localization(args, pred_meta_topk, pred_pose)

    print("[i] cross-view image geo-localization finished")
    print("[i] query path: ", args["qry_path"])
    print("[i] estimation result: Lat: ", pred_locs[0][0], " Long: ", pred_locs[0][1])

    output_dir = "./output"
    save_output(args, pred_meta_topk, pred_locs, output_dir)

    return


def parse_args():
    parser = argparse.ArgumentParser(description="CIPER inference")
    parser.add_argument("--config", help="config file path", required=True)
    # parser.add_argument("--query", help="query file path", default="", required=True)
    args = parser.parse_args()
    return args


def main():
    cmd_args = parse_args()

    global args
    with open(cmd_args.config, "r") as stream:
        args = yaml.safe_load(stream)
    
    while True:
        query_path = prompt("query file path(ex: demo/demo_sample_1.png): ", completer=PathCompleter())
        if not os.path.exists(query_path):
            continue
        if query_path.split('.')[-1] not in ["png", "jpg"]:
            continue
        args["qry_path"] = query_path
        inference()

    if cmd_args.query != "":
        args["qry_path"] = cmd_args.query

    return


if __name__ == "__main__":
    main()
