"""
Cleaned & modernized train/eval script for CIPER.

What this refactor does (keeping your existing engine/datasets/utils API):
- Clear separation: config/load, seed, build model, load weights, optimizer, loaders, loop
- Modern ViT training knobs (AMP, grad clip, EMA optional hooks)
- Safer pretrained loading (skip pos_embed/patch_embed by default)
- Resume handling unified
- W&B init guarded by debug

Assumptions:
- build(args) -> (model, criterion, postprocessors)
- build_dataset(mode, args) exists
- engine: train_one_epoch / valid_one_epoch / evaluate_one
- common.utils: adjust_learning_rate, save_state, print_pigeon_train, print_pigeon_evaluation
"""

import argparse
import os
import random
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.append("../")

import wandb
from common.utils import (adjust_learning_rate, print_pigeon_evaluation,
                          print_pigeon_train, save_state)
from datasets import build_dataset
from engine import evaluate_one, train_one_epoch, valid_one_epoch
from models import SAM, build


# -----------------------------
# utils
# -----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Determinism: can slow down; make configurable
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def count_trainable_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def maybe_init_wandb(args: Dict[str, Any], debug: bool, run_name: str) -> None:
    if debug:
        return
    wandb.init(
        project=args.get("wandb_project", "ciper-v3"),
        config=args,
        name=run_name,
        settings=wandb.Settings(start_method="fork"),
    )
    wandb.run.name = f"{run_name}-{wandb.run.id}"


def load_pretrained_partial(
    model: torch.nn.Module,
    ckpt_path: str,
    *,
    skip_keys=("pos_embed", "patch_embed"),
    strip_module_prefix: bool = True,
    freeze_loaded: bool = True,
) -> None:
    """
    Loads checkpoint["state_dict"] partially:
      - skips keys containing any of skip_keys
      - strips 'module.' prefix (DDP)
      - optionally freezes loaded params
    """
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint.get("model", None))
    if state_dict is None:
        raise ValueError(
            f"Checkpoint at {ckpt_path} has no 'state_dict' or 'model' key."
        )

    new_state = {}
    for k, v in state_dict.items():
        if any(s in k for s in skip_keys):
            continue
        if strip_module_prefix and k.startswith("module."):
            k = k[len("module.") :]
        new_state[k] = v

    msg = model.load_state_dict(new_state, strict=False)
    print("[i] partial pretrained load:", msg)

    if freeze_loaded:
        loaded_keys = set(new_state.keys())
        for name, param in model.named_parameters():
            if name in loaded_keys:
                param.requires_grad = False
        print(f"[i] froze {len(loaded_keys)} loaded tensors")


def load_resume(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    ckpt_path: str,
) -> int:
    """
    Loads a full training checkpoint:
      - expects checkpoint['model']
      - optionally loads optimizer and epoch if present
    Returns: start_epoch
    """
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if "model" not in checkpoint:
        raise ValueError(f"Resume checkpoint must contain 'model': {ckpt_path}")

    model.load_state_dict(checkpoint["model"], strict=True)

    start_epoch = 0
    if optimizer is not None and "optimizer" in checkpoint and "epoch" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        print(f"[i] resume from {ckpt_path} (epoch={checkpoint['epoch']})")
    else:
        print(f"[i] loaded weights from {ckpt_path} (no optimizer/epoch found)")

    return start_epoch


def build_optimizer(args: Dict[str, Any], model: torch.nn.Module):
    param_list = [p for p in model.parameters() if p.requires_grad]
    opt = args.get("optimizer", "adamw").lower()

    lr = float(args["lr"])
    wd = float(args.get("weight_decay", 0.0))

    if opt == "adam":
        return torch.optim.Adam(
            param_list, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=wd
        )
    if opt == "adamw":
        return torch.optim.AdamW(
            param_list, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=wd
        )
    if opt == "sam":
        base_optimizer = torch.optim.AdamW
        return SAM(
            param_list,
            base_optimizer,
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=wd,
            amsgrad=False,
            rho=float(args.get("sam_rho", 2.5)),
            adaptive=bool(args.get("sam_adaptive", True)),
        )
    raise ValueError(f"Unknown optimizer: {opt}")


def build_train_loader(args: Dict[str, Any]) -> DataLoader:
    dataset_train = build_dataset(mode="train", args=args)
    return DataLoader(
        dataset_train,
        batch_size=int(args["batch_size"]),
        shuffle=True,
        num_workers=int(args["num_workers"]),
        pin_memory=True,
        drop_last=True,
    )


def build_valid_loaders(args: Dict[str, Any]) -> Dict[str, Dict[str, DataLoader]]:
    """
    Returns:
      loaders = {
        "same": {"qry": ..., "ref": ..., "val": ...},
        "cross": {"qry": ..., "ref": ..., "val": ...}  # if kitti
      }
    """
    num_workers = int(args["num_workers"])

    # retrieval validation (same)
    ds_s_q = build_dataset(mode="valid_same_qry", args=args)
    ds_s_r = build_dataset(mode="valid_same_ref", args=args)
    loaders_same = {
        "qry": DataLoader(
            ds_s_q,
            batch_size=int(args.get("val_qry_bs", 32)),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "ref": DataLoader(
            ds_s_r,
            batch_size=int(args.get("val_ref_bs", 64)),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    }

    # pose validation (same)
    ds_val_same = build_dataset(mode="valid_same", args=args)
    loaders_same["val"] = DataLoader(
        ds_val_same,
        batch_size=int(args["batch_size"]),
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    loaders = {"same": loaders_same}

    if args.get("data_name", "") == "kitti":
        ds_c_q = build_dataset(mode="valid_cross_qry", args=args)
        ds_c_r = build_dataset(mode="valid_cross_ref", args=args)
        loaders_cross = {
            "qry": DataLoader(
                ds_c_q,
                batch_size=int(args.get("val_qry_bs", 32)),
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            ),
            "ref": DataLoader(
                ds_c_r,
                batch_size=int(args.get("val_ref_bs", 64)),
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            ),
        }
        ds_val_cross = build_dataset(mode="valid_cross", args=args)
        loaders_cross["val"] = DataLoader(
            ds_val_cross,
            batch_size=int(args["batch_size"]),
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
        )
        loaders["cross"] = loaders_cross

    return loaders


# -----------------------------
# train / eval
# -----------------------------
def train_loop(args: Dict[str, Any], debug: bool) -> None:
    # seed
    set_seed(int(args.get("seed", 42)))

    # build
    model, criterion, postprocessors = build(args)

    # wandb
    run_name = (
        os.path.splitext(os.path.basename(sys.argv[2]))[0]
        if len(sys.argv) > 2
        else "ciper_run"
    )
    maybe_init_wandb(args, debug, run_name)

    # pretrained / resume
    # priority: resume > pretrained-partial
    optimizer = build_optimizer(args, model)

    start_epoch = int(args.get("start_epoch", 1))
    if args.get("resume", False):
        start_epoch = load_resume(model, optimizer, args["resume"])
        args["start_epoch"] = start_epoch
    elif args.get("pretrained", False):
        load_pretrained_partial(
            model,
            args["pretrained"],
            skip_keys=tuple(
                args.get("pretrained_skip_keys", ["pos_embed", "patch_embed"])
            ),
            strip_module_prefix=True,
            freeze_loaded=bool(args.get("freeze_pretrained", True)),
        )

    n_params = count_trainable_params(model)
    print("[i] trainable params:", n_params // 10**6, "M")

    # loaders
    train_loader = build_train_loader(args)
    valid_loaders = build_valid_loaders(args)

    # infos
    train_infos = {
        "iter": 0,
        "epoch": -1,
        "device": args["device"],
        "optimizer": args.get("optimizer", "adamw"),
        # Modern recipe knobs forwarded to engine if you support them:
        "use_amp": bool(args.get("use_amp", True)),
        "clip_max_norm": float(args.get("clip_max_norm", 0.0)),  # 0 disables
    }
    valid_infos = {
        "epoch": -1,
        "device": args["device"],
        "best_metric": -1,
        "dim_feature": args["dim_feature"],
    }

    # AMP scaler (if your engine supports; otherwise harmless to keep here)
    # You can pass this into engine if you want.
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.get("use_amp", True)))
    train_infos["scaler"] = scaler

    print("[i] start training ~")
    for epoch in range(int(args["start_epoch"]), int(args["epochs"]) + 1):
        adjust_learning_rate(optimizer, epoch, args)

        train_infos["epoch"] = epoch
        train_infos = train_one_epoch(
            model, criterion, postprocessors, train_loader, optimizer, train_infos
        )

        valid_infos["epoch"] = epoch

        # valid: same
        valid_infos = valid_one_epoch(
            model,
            criterion,
            postprocessors,
            valid_loaders["same"],
            ({**valid_infos, **dict(valid="same")}),
        )

        # valid: cross (kitti)
        if "cross" in valid_loaders:
            valid_infos = valid_one_epoch(
                model,
                criterion,
                postprocessors,
                valid_loaders["cross"],
                ({**valid_infos, **dict(valid="cross")}),
            )

        # best checkpoint by 'metric'
        is_best = False
        if valid_infos.get("metric", -1) > valid_infos["best_metric"]:
            valid_infos["best_metric"] = valid_infos["metric"]
            is_best = True

        save_state(model, optimizer, epoch, is_best)

    if wandb.run is not None:
        wandb.finish()


@torch.no_grad()
def eval_loop(args: Dict[str, Any]) -> None:
    set_seed(int(args.get("seed", 42)))

    model, _, postprocessors = build(args)
    n_params = count_trainable_params(model)
    print("[i] trainable params:", n_params // 10**6, "M")

    ckpt_path = args["pretrained"]
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    print("[i] load checkpoint from:", ckpt_path, "for evaluation")

    valid_loaders = build_valid_loaders(args)

    eval_infos = {
        "device": args["device"],
        "dim_feature": args["dim_feature"],
        "data_name": args["data_name"],
        "eval_name": args.get("eval_name", "eval"),
    }

    evaluate_one(
        model,
        postprocessors,
        valid_loaders["same"],
        ({**eval_infos, **dict(valid="same")}),
    )

    if "cross" in valid_loaders:
        evaluate_one(
            model,
            postprocessors,
            valid_loaders["cross"],
            ({**eval_infos, **dict(valid="cross")}),
        )


# -----------------------------
# cli
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train/Eval CIPER")
    p.add_argument("--config", required=True, help="config file path")
    p.add_argument("--debug", action="store_true", help="disable logger")
    return p.parse_args()


def main():
    cmd = parse_args()
    with open(cmd.config, "r") as f:
        args = yaml.safe_load(f)

    if args.get("eval", False):
        print_pigeon_evaluation()
        eval_loop(args)
    else:
        print_pigeon_train()
        train_loop(args, cmd.debug)


if __name__ == "__main__":
    main()
