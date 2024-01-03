import os, sys, yaml
import argparse
import torch
from torch.utils.data import DataLoader

import sys

sys.path.append("../")

from common.utils import print_pigeon
from datasets import build_dataset
from models import build, SAM
from engine import train_one_epoch, valid_one_epoch, evaluate

import wandb


def parse_args():
    parser = argparse.ArgumentParser(description="Train a CIPER")
    parser.add_argument("config", help="config file path")
    parser.add_argument(
        "--debug", action="store_true", help="debug flag for disble logger"
    )
    args = parser.parse_args()
    return args


def adjust_learning_rate(optimizer, epoch, args):
    import math

    """Decay the learning rate based on schedule"""
    lr = args["lr"]
    if args["cos"]:  # cosine lr schedule
        lr *= 0.5 * (1.0 + math.cos(math.pi * epoch / args["epochs"]))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.0
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def save_state(model, optimizer, epoch, is_best):
    # os.makedirs(save_path, exist_ok=True)
    state_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }
    # save_name = os.path.join(save_path, "epoch_" + str(epoch)+".pth")
    # torch.save(state_dict, save_name)
    # print("[i] checkpoint saved in ", save_name)

    # if is_best:
    #     torch.save(state_dict, os.path.join(save_path, "model_best.pth"))
    #     print("[i] best checkpoint saved in ", os.path.join(save_path, "model_best.pth"))
    # if epoch > 3:
    #     prev_checkpoint_filename = os.path.join(
    #         save_path, "epoch_" + str(epoch - 3) + ".pth")
    #     if os.path.exists(prev_checkpoint_filename):
    #         os.remove(prev_checkpoint_filename)
    if wandb.run is not None:
        save_name = os.path.join(wandb.run.dir, "epoch_" + str(epoch) + ".pth")
        torch.save(state_dict, save_name)
        # wandb.save(save_name)


def main():
    cmd_args = parse_args()

    global args
    with open(cmd_args.config, "r") as stream:
        args = yaml.safe_load(stream)

    print_pigeon()

    model, criterion, postprocessors = build(args)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("[i] number of params:", n_parameters // 10**6, "M")

    if not cmd_args.debug:
        wandb.init(
            # set the wandb project where this run will be logged
            project="CIPER",
            config=args,
            name=sys.argv[1].split("/")[-1].split(".")[0],
            resume=(args["resume"] != False),
        )

    if args["resume"] != False:
        checkpoint = torch.load(args["resume"], map_location="cpu")
        print(checkpoint.keys())
        model.load_state_dict(checkpoint["model"])
        if args["infer"]:
            print("[i] load checkpoint from:", args["resume"], "for inference")
        elif "optimizer" in checkpoint and "epoch" in checkpoint:
            args["start_epoch"] = checkpoint["epoch"] + 1
            print("[i] load checkpoint from:", args["resume"], "for train")
        else:
            print("[i] failed to load checkpoint from:", args["resume"])
            return

    if not args["infer"]:
        param_dicts = list(filter(lambda p: p.requires_grad, model.parameters()))

        # optimizer and ir_scheduler
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

        ## data_loader
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

    data_loader_valid_same = {"qry": data_loader_val_s_q, "ref": data_loader_val_s_r}

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

    if not args["retr_only"]:
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

    if args["infer"]:
        eval_infos = {
            "device": args["device"],
            "retr_only": args["retr_only"],
            "dim_feature": args["dim_feature"],
        }
        evaluate(
            model,
            postprocessors,
            data_loader_valid_same,
            (eval_infos | dict(valid="same")),
        )
        evaluate(
            model,
            postprocessors,
            data_loader_valid_cross,
            (eval_infos | dict(valid="cross")),
        )
        return

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

        if args["data_name"] == "kitti":
            valid_infos = valid_one_epoch(
                model,
                criterion,
                postprocessors,
                data_loader_valid_same,
                ({**valid_infos, **dict(valid="same")}),
            )
            valid_infos = valid_one_epoch(
                model,
                criterion,
                postprocessors,
                data_loader_valid_cross,
                ({**valid_infos, **dict(valid="cross")}),
            )
        elif args["data_name"] == "ford":
            valid_infos = valid_one_epoch(
                model,
                criterion,
                postprocessors,
                data_loader_valid_same,
                ({**valid_infos, **dict(valid="same")}),
            )

        is_best = False
        if valid_infos["metric"] > valid_infos["best_metric"]:
            valid_infos["best_metric"] = valid_infos["metric"]
            is_best = True

        save_state(model, optimizer, epoch, is_best)

    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
