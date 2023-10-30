import os, sys, yaml
import datetime

import torch
from torch.utils.data import DataLoader

import shutil

import sys; sys.path.append("../")

from common.utils import save_state, print_pigeon
from datasets import build_dataset
from models import build, SAM
from engine import train_one_epoch, valid_one_epoch, evaluate

import wandb

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
    
    wandb.init(
        # set the wandb project where this run will be logged
        project="CIPER",
        config=args,
        name=sys.argv[1].split('/')[-1].split('.')[0]
    )

    print_pigeon()

    model, criterion, postprocessors = build(args)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("[i] number of params:", n_parameters // 10 ** 6, "M")
    
    if args["resume"] != False:
        checkpoint = torch.load(args["resume"], map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        if args["infer"]:
            print("[i] load checkpoint from:", args["resume"], "for inference")
        elif "optimizer" in checkpoint and "lr_scheduler" in checkpoint and "epoch" in checkpoint:
            args["start_epoch"] = checkpoint["epoch"] + 1        
            print("[i] load checkpoint from:", args["resume"], "for train")
        else:
            print("[i] failed to load checkpoint from:", args["resume"])
            return  
    
    if not args["infer"] :
        
        param_dicts = list(filter(lambda p: p.requires_grad, model.parameters()))
        
        # optimizer and ir_scheduler         
        if args["optimizer"] == "adam":            
            optimizer = torch.optim.Adam(param_dicts, lr=args["lr"], betas=(0.9, 0.999), eps=1e-08, 
                                        weight_decay=args["weight_decay"])
        elif args["optimizer"] == "adamw":            
            optimizer = torch.optim.AdamW(param_dicts, lr=args["lr"], betas=(0.9, 0.999), eps=1e-08, 
                                        weight_decay=args["weight_decay"], amsgrad=False)
        elif args["optimizer"] == "sam":            
            base_optimizer = torch.optim.AdamW
            optimizer = SAM(param_dicts, base_optimizer, lr=args["lr"], betas=(0.9, 0.999), eps=1e-08, 
                            weight_decay=args["weight_decay"], amsgrad=False, rho=2.5, adaptive=True)
        
        ## data_loader  
        dataset_train = build_dataset(mode="train", args=args)
        dataset_val_q = build_dataset(mode="valid_qry", args=args)
        dataset_val_r = build_dataset(mode="valid_ref", args=args)

        sampler_train = None
            
        data_loader_train = DataLoader(dataset_train, batch_size=args["batch_size"], 
                                    shuffle=(sampler_train is None), 
                                    num_workers=args["num_workers"], pin_memory=True, sampler=sampler_train, drop_last=True)
        data_loader_val_q = DataLoader(dataset_val_q, batch_size=32, shuffle=False,
                                        num_workers=args["num_workers"], pin_memory=True) 
        data_loader_val_r = DataLoader(dataset_val_r, batch_size=64, shuffle=False,
                                        num_workers=args["num_workers"], pin_memory=True)
        
        data_loader_valid = {
            "qry": data_loader_val_q,
            "ref": data_loader_val_r
        }
            
        if not args["retr_only"]:            
            dataset_val = build_dataset(mode="valid", args=args)
            data_loader_val = DataLoader(dataset_val, batch_size=args["batch_size"], shuffle=False, drop_last=False,
                                        num_workers=args["num_workers"]) 
            data_loader_valid["val"] = data_loader_val
            
        if args["data_name"] == "kitti":
            dataset_val_q2 = build_dataset(mode="valid2_qry", args=args)
            dataset_val_r2 = build_dataset(mode="valid2_ref", args=args)
            data_loader_val_q2 = DataLoader(dataset_val_q2, batch_size=32, shuffle=False,
                                        num_workers=args["num_workers"], pin_memory=True) 
            data_loader_val_r2 = DataLoader(dataset_val_r2, batch_size=64, shuffle=False,
                                        num_workers=args["num_workers"], pin_memory=True)
            data_loader_valid["qry2"] = data_loader_val_q2
            data_loader_valid["ref2"] = data_loader_val_r2
            if not args["retr_only"]:       
                dataset_val2 = build_dataset(mode="valid2", args=args)
                data_loader_val2 = DataLoader(dataset_val2, batch_size=args["batch_size"], shuffle=False, drop_last=False,
                                            num_workers=args["num_workers"]) 
                data_loader_valid["val2"] = data_loader_val2
        
        out_dir = os.path.join(args["ckpt_dir"], 
                            args["data_name"] + "-" + datetime.datetime.today().strftime("%d-%m-%y-%H:%M:%S"))
        os.makedirs(out_dir)
        shutil.copyfile(sys.argv[1], os.path.join(out_dir, "config.yaml"))  
            
    else:
        dataset_val_q = build_dataset(mode="valid_qry", args=args)
        dataset_val_r = build_dataset(mode="valid_ref", args=args)

        data_loader_val_q = DataLoader(dataset_val_q, batch_size=32, shuffle=False,
                                        num_workers=args["num_workers"], pin_memory=True) 
        data_loader_val_r = DataLoader(dataset_val_r, batch_size=64, shuffle=False,
                                        num_workers=args["num_workers"], pin_memory=True)
        
        data_loader_valid = {
            "qry": data_loader_val_q,
            "ref": data_loader_val_r
        }
        
        if not args["retr_only"]:         
            dataset_val = build_dataset(mode="valid", args=args)
            data_loader_val = DataLoader(dataset_val, batch_size=args["batch_size"], shuffle=False, drop_last=False,
                                        num_workers=args["num_workers"]) 
            data_loader_valid["val"] = data_loader_val

    if args["infer"]:
        eval_infos = {
        "device": args["device"],
        "retr_only": args["retr_only"],
        "dim_feature": args["dim_feature"],        
        }        
        evaluate(model, postprocessors, data_loader_valid, eval_infos)
        return

    print("[i] start training ~")
    train_infos = {
        "iter" : 0,
        "epoch": -1,
        "device": args["device"],
        "retr_only": args["retr_only"],
        # "clip_max_norm": args["clip_max_norm"],
        "optimizer": args["optimizer"]
    }
    valid_infos = {
        "epoch": -1,
        "device": args["device"],
        "retr_only": args["retr_only"],
        "best_metric": -1,
        "dim_feature": args["dim_feature"],     
        "data_name": args["data_name"],        
    }

    # print(len(data_loader_valid["qry"].dataset), len(data_loader_valid["ref"].dataset)); exit()
        
    for epoch in range(args["start_epoch"], args["epochs"] + 1):
        
        adjust_learning_rate(optimizer, epoch, args)

        # train_infos["epoch"] = epoch
        # train_infos = train_one_epoch(
        #         model, criterion, postprocessors, data_loader_train, optimizer, train_infos)
            
        valid_infos["epoch"] = epoch
        valid_infos = valid_one_epoch(model, criterion, postprocessors, data_loader_valid, valid_infos)   
        
        is_best = False
        if valid_infos["metric"] > valid_infos["best_metric"]:
            valid_infos["best_metric"] = valid_infos["metric"]
            is_best = True       
            
        save_state(out_dir, model, optimizer, epoch, is_best)

    wandb.finish()
        
if __name__ == "__main__":
    main()
