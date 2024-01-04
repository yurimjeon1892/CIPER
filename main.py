import os, sys, yaml
import argparse
import torch
from torch.utils.data import DataLoader

import sys

sys.path.append("../")

from common.utils import print_pigeon, adjust_learning_rate, save_state
from datasets import build_dataset
from models import build, SAM
from engine import train_one_epoch, valid_one_epoch, evaluate_one

import wandb


def parse_args():
    parser = argparse.ArgumentParser(description="Train a CIPER")
    parser.add_argument("config", help="config file path")
    parser.add_argument(
        "--debug", action="store_true", help="debug flag for disble logger"
    )
    args = parser.parse_args()
    return args


def iterate(debug):
    ## init model
    model, criterion, postprocessors = build(args)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("[i] number of params:", n_parameters // 10**6, "M")

    ## init wandb
    if not debug:
        wandb.init(
            # set the wandb project where this run will be logged
            project="CIPER",
            config=args,
            name=sys.argv[1].split("/")[-1].split(".")[0],
            resume=(args["resume"] != False),
        )

    ## resume model
    if args["resume"] != False:
        checkpoint = torch.load(args["resume"], map_location="cpu")
        print(checkpoint.keys())
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint and "epoch" in checkpoint:
            args["start_epoch"] = checkpoint["epoch"] + 1
            print("[i] load checkpoint from:", args["resume"], "for train")
        else:
            print("[i] failed to load checkpoint from:", args["resume"])
            return

    ## set optimizer and ir_scheduler
    param_dicts = list(filter(lambda p: p.requires_grad, model.parameters()))
    if args["optimizer"] == "adam":
        optimizer = torch.optim.Adam(
            param_dicts,
            lr=args["lr"],
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args["weight_decay"],
        )
    elif args["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(
            param_dicts,
            lr=args["lr"],
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args["weight_decay"],
            amsgrad=False,
        )
    elif args["optimizer"] == "sam":
        base_optimizer = torch.optim.AdamW
        optimizer = SAM(
            param_dicts,
            base_optimizer,
            lr=args["lr"],
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args["weight_decay"],
            amsgrad=False,
            rho=2.5,
            adaptive=True,
        )

    ## set data_loader for train
    dataset_train = build_dataset(mode="train", args=args)

    sampler_train = None
    data_loader_train = DataLoader(
        dataset_train,
        batch_size=args["batch_size"],
        shuffle=(sampler_train is None),
        num_workers=args["num_workers"],
        pin_memory=True,
        sampler=sampler_train,
        drop_last=True,
    )

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

    ## set infos for train / validation
    print("[i] start training ~")
    train_infos = {
        "iter": 0,
        "epoch": -1,
        "device": args["device"],
        "retr_only": args["retr_only"],
        # "clip_max_norm": args["clip_max_norm"],
        "optimizer": args["optimizer"],
    }
    valid_infos = {
        "epoch": -1,
        "device": args["device"],
        "retr_only": args["retr_only"],
        "best_metric": -1,
        "dim_feature": args["dim_feature"],
    }

    # print(len(data_loader_valid["qry"].dataset), len(data_loader_valid["ref"].dataset)); exit()

    for epoch in range(args["start_epoch"], args["epochs"] + 1):
        adjust_learning_rate(optimizer, epoch, args)

        train_infos["epoch"] = epoch
        train_infos = train_one_epoch(
            model, criterion, postprocessors, data_loader_train, optimizer, train_infos
        )

        valid_infos["epoch"] = epoch

        valid_infos = valid_one_epoch(
            model,
            criterion,
            postprocessors,
            data_loader_valid_same,
            ({**valid_infos, **dict(valid="same")}),
        )

        if args["data_name"] == "kitti":
            valid_infos = valid_one_epoch(
                model,
                criterion,
                postprocessors,
                data_loader_valid_cross,
                ({**valid_infos, **dict(valid="cross")}),
            )

        is_best = False
        if valid_infos["metric"] > valid_infos["best_metric"]:
            valid_infos["best_metric"] = valid_infos["metric"]
            is_best = True

        save_state(model, optimizer, epoch, is_best)

    if wandb.run is not None:
        wandb.finish()

    return


def evaluate():
    ## init model
    model, _, postprocessors = build(args)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("[i] number of params:", n_parameters // 10**6, "M")

    ## load pretrained
    checkpoint = torch.load(args["pretrained"], map_location="cpu")
    print(checkpoint.keys())
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
    print("[i] start evaluation ~")

    eval_infos = {
        "device": args["device"],
        "dim_feature": args["dim_feature"],
    }
    evaluate(
        model,
        postprocessors,
        data_loader_valid_same,
        (eval_infos | dict(valid="same")),
    )
    if args["data_name"] == "kitti":
        evaluate(
            model,
            postprocessors,
            data_loader_valid_cross,
            (eval_infos | dict(valid="cross")),
        )
    return


def main():
    cmd_args = parse_args()

    global args
    with open(cmd_args.config, "r") as stream:
        args = yaml.safe_load(stream)

    print_pigeon()

    if args["eval"]:
        evaluate()
    else:
        iterate(cmd_args.debug)

    return


if __name__ == "__main__":
    main()
