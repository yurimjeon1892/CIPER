# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os, sys, yaml
import random
import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter

import shutil

import sys; sys.path.append("../")
import common.utils_misc as utils_misc

from common.utils import save_state, print_pigeon
from datasets import build_dataset
from models import build, SAM
from engine import train_one_epoch, valid_one_epoch, evaluate

def adjust_learning_rate(optimizer, epoch, args):
    import math
    """Decay the learning rate based on schedule"""
    lr = args["lr"]
    if args["cos"]:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args["epochs"]))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def main():
    
    global args
    with open(sys.argv[1], "r") as stream:        
        args = yaml.safe_load(stream)      
                      
    print_pigeon()

    device = torch.device(args["device"])
    
    # # fix the seed for reproducibility
    # random.seed(args["seed"])
    # torch.manual_seed(args["seed"])
    # # np.random.seed(args["seed"])    
    
    IS_POSE = args["task"] == "POSE"

    model, criterion, postprocessors = build(args["model"], IS_POSE, device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("[i] number of params:", n_parameters // 10 ** 6, "M")
    
    if args["resume"] != False:
        checkpoint = torch.load(args["resume"], map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        if args["infer"]:
            print("[i] load checkpoint from:", args["resume"], "for inference")
        elif "optimizer" in checkpoint and "lr_scheduler" in checkpoint and "epoch" in checkpoint:
            args["train"]["start_epoch"] = checkpoint["epoch"] + 1        
            print("[i] load checkpoint from:", args["resume"], "for train")
        else:
            print("[i] failed to load checkpoint from:", args["resume"])
            return  
    
    if not args["infer"] :
        
        param_dicts = list(filter(lambda p: p.requires_grad, model.parameters()))
        
        # optimizer and ir_scheduler         
        if args["train"]["optimizer"] == "adam":            
            optimizer = torch.optim.Adam(param_dicts, lr=args["train"]["lr"], betas=(0.9, 0.999), eps=1e-08, 
                                        weight_decay=args["train"]["weight_decay"])
        elif args["train"]["optimizer"] == "adamw":            
            optimizer = torch.optim.AdamW(param_dicts, lr=args["train"]["lr"], betas=(0.9, 0.999), eps=1e-08, 
                                        weight_decay=args["train"]["weight_decay"], amsgrad=False)
        elif args["train"]["optimizer"] == "sam":            
            base_optimizer = torch.optim.AdamW
            optimizer = SAM(param_dicts, base_optimizer, lr=args["train"]["lr"], betas=(0.9, 0.999), eps=1e-08, 
                            weight_decay=args["train"]["weight_decay"], amsgrad=False, rho=2.5, adaptive=True)

        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args["train"]["lr_drop"])
        
        ## data_loader  
        dataset_train = build_dataset(mode="train", args=args["dataset"])
        dataset_val_q = build_dataset(mode="valid_qry", args=args["dataset"])
        dataset_val_r = build_dataset(mode="valid_ref", args=args["dataset"])

        sampler_train = None
            
        data_loader_train = DataLoader(dataset_train, batch_size=args["batch_size"], shuffle=(sampler_train is None), sampler=sampler_train,  drop_last=True,
                                        num_workers=args["num_workers"]) 
        data_loader_val_q = DataLoader(dataset_val_q, batch_size=32, shuffle=True,
                                        drop_last=False, num_workers=args["num_workers"]) 
        data_loader_val_r = DataLoader(dataset_val_r, batch_size=64, shuffle=True,
                                        drop_last=False, num_workers=args["num_workers"]) 
        
        data_loader_valid = {
            "qry": data_loader_val_q,
            "ref": data_loader_val_r
        }
            
        if IS_POSE:            
            dataset_val = build_dataset(mode="valid", args=args["dataset"])
            data_loader_val = DataLoader(dataset_val, batch_size=args["batch_size"], shuffle=False, drop_last=False,
                                        num_workers=args["num_workers"]) 
            data_loader_valid["val"] = data_loader_val
        
        out_dir = os.path.join(args["train"]["ckpt_dir"], 
                               args["dataset"]["data_name"] + "-" + datetime.datetime.today().strftime("%d-%m-%y-%H:%M:%S"))
        summary = SummaryWriter(out_dir, "tb")
        shutil.copyfile(sys.argv[1], os.path.join(out_dir, "config.yaml"))  
            
    else:
        dataset_val_q = build_dataset(mode="valid_qry", args=args["dataset"])
        dataset_val_r = build_dataset(mode="valid_ref", args=args["dataset"])

        data_loader_val_q = DataLoader(dataset_val_q, batch_size=32, shuffle=False,
                                        drop_last=False, num_workers=args["num_workers"])
        data_loader_val_r = DataLoader(dataset_val_r, batch_size=64, shuffle=False,
                                        drop_last=False, num_workers=args["num_workers"])
        
        data_loader_valid = {
            "qry": data_loader_val_q,
            "ref": data_loader_val_r
        }
        
        if IS_POSE:            
            dataset_val = build_dataset(mode="valid", args=args["dataset"])
            data_loader_val = DataLoader(dataset_val, batch_size=args["batch_size"], shuffle=False, drop_last=False,
                                        num_workers=args["num_workers"]) 
            data_loader_valid["val"] = data_loader_val

    if args["infer"]:
        eval_infos = {
        "device": device,
        "IS_POSE": IS_POSE,
        "dim_feature": args["model"]["dim_feature"],        
        }        
        evaluate(model, criterion, postprocessors, data_loader_valid, eval_infos)
        return

    print("[i] start training ~")
    train_infos = {
        "iter" : 0,
        "epoch": -1,
        "device": device,
        "IS_POSE": IS_POSE,
        # "clip_max_norm": args["train"]["clip_max_norm"],
        "optimizer": args["train"]["optimizer"]
    }
    valid_infos = {
        "epoch": -1,
        "device": device,
        "best_metric": -1,
        "IS_POSE": IS_POSE,
        "dim_feature": args["model"]["dim_feature"],        
    }
     
    for epoch in range(args["train"]["start_epoch"], args["train"]["epochs"] + 1):
        
        adjust_learning_rate(optimizer, epoch, args["train"])

        train_infos["epoch"] = epoch
        train_infos = train_one_epoch(
                model, criterion, postprocessors, data_loader_train, optimizer, train_infos, summary)
        # lr_scheduler.step() 
            
        valid_infos["epoch"] = epoch
        valid_infos = valid_one_epoch(model, criterion, postprocessors, data_loader_valid, valid_infos, summary)   
        
        is_best = False
        if valid_infos["metric"] > valid_infos["best_metric"]:
            valid_infos["best_metric"] = valid_infos["metric"]
            is_best = True       
            
        save_state(out_dir, model, optimizer, None, epoch, is_best)
        
if __name__ == "__main__":
    main()
