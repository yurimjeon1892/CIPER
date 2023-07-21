# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os, sys, yaml
import random
import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

from tensorboardX import SummaryWriter

import shutil

import sys; sys.path.append("../")
import common.utils_misc as utils_misc

from common.utils import save_state, load_pretrained, print_pigeon
from datasets import build_dataset
from models import build
from engine import train_one_epoch, valid_one_epoch, evaluate

def main():
    
    # parse arguments
    global args
    with open(sys.argv[1], "r") as stream:        
        args = yaml.safe_load(stream)      
                      
    print_pigeon()
                
    # utils_misc.init_distributed_mode(args) # Multi-GPU 사용할 거라면, args.gpu / args.world_size / args.rank 가 여기서 정의 된다.

    device = torch.device(args["device"])
    
    # Multi-GPU 사용할 거라면, fix the seed for reproducibility 
    # fix the seed for reproducibility
    seed = args["seed"] + utils_misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    is_local = args["task"] == "LOCAL"

    model, criterion, postprocessors = build(args["model"], is_local, device)

    # model_without_ddp = model
    # if args["distributed"]:
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args["gpu"]])
    #     model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # for name, param in model.named_parameters():
    #     print(name)    
    print("[i] number of params:", n_parameters // 10 ** 6 , "M")
    
    if not args["infer"] :
        
        ## backbone / Transformer-encoder, decoder / detector head 각각의 learning rate를 다르게 주는 방법
        param_dicts = [
            {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": args["train"]["lr_backbone"],
            },
        ]
        
        ## optimizer와 ir_scheduler 설정
        # optimizer = torch.optim.Adam(param_dicts, lr=args["train"]["lr"], betas=(0.9, 0.999), eps=1e-08, 
        #                             weight_decay=args["train"]["weight_decay"])
        optimizer = torch.optim.AdamW(param_dicts, lr=args["train"]["lr"], betas=(0.9, 0.999), eps=1e-08, 
                                      weight_decay=args["train"]["weight_decay"], amsgrad=False)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args["train"]["lr_drop"], args["train"]["gamma"])
        
        ## data_loader 만들어 주기 
        # train -> dataset -> RandomSampler -> BatchSampler -> DataLoader
        # val -> dataset -> SequentialSampler -> DataLoader(+batch_size)
        dataset_train = build_dataset(mode="train", args=args["dataset"])
        dataset_val_q = build_dataset(mode="valid_qry", args=args["dataset"])
        dataset_val_r = build_dataset(mode="valid_ref", args=args["dataset"])

        # if args["distributed"]: sampler_train = DistributedSampler(dataset_train)
        # else: sampler_train = None
        sampler_train = None
            
        # # data_loader에서는 1장씩만 뱉어주면 된다. BatchSampler가 Batch로 묶어 준다.
        # batch_sampler_train = torch.utils_misc.data.BatchSampler(sampler_train, args["batch_size"], drop_last=True)
        
        # 특히 data_loader_train에서 batch_size를 정의하지 않고, BatchSampler라는 함수를 사용했다.
        # utils_misc.collate_fn 함수에 의해서, (image, label) -> (NestedTensor(tensor,mask), label) 로 바뀐다
        data_loader_train = DataLoader(dataset_train, batch_size=args["batch_size"], shuffle=(sampler_train is None), sampler=sampler_train,  drop_last=True,
                                        collate_fn=utils_misc.collate_fn, num_workers=args["num_workers"]) 
        # data_loader_train = DataLoader(dataset_train, batch_size=args["batch_size"], shuffle=False, sampler=sampler_train,  drop_last=True,
        #                                 collate_fn=utils_misc.collate_fn, num_workers=args["num_workers"]) 
        data_loader_val_q = DataLoader(dataset_val_q, batch_size=32, shuffle=False,
                                        drop_last=False, num_workers=args["num_workers"])
        data_loader_val_r = DataLoader(dataset_val_r, batch_size=64, shuffle=False,
                                        drop_last=False, num_workers=args["num_workers"])
        
        data_loader_valid = {
            "qry": data_loader_val_q,
            "ref": data_loader_val_r
        }
            
        if is_local:            
            dataset_val = build_dataset(mode="valid", args=args["dataset"])
            data_loader_val = DataLoader(dataset_val, batch_size=args["batch_size"], shuffle=False, drop_last=True,
                                        collate_fn=utils_misc.collate_fn, num_workers=args["num_workers"]) 
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
        
        if is_local:            
            dataset_val = build_dataset(mode="valid", args=args["dataset"])
            data_loader_val = DataLoader(dataset_val, batch_size=args["batch_size"], shuffle=False, drop_last=True,
                                        collate_fn=utils_misc.collate_fn, num_workers=args["num_workers"]) 
            data_loader_valid["val"] = data_loader_val

    # TODO ##########################################################################################################
    #                                                                                                               #
    #  
    if args["pretrain"] != False:
        model = load_pretrained(model, args["pretrain"])
        print("[i] load pretrained file from:", args["pretrain"])
    if args["resume"]  != False:
        checkpoint = torch.load(args["resume"], map_location="cpu")
        # model_without_ddp.load_state_dict(checkpoint["model"])
        model.load_state_dict(checkpoint["model"])
        if args["infer"]:
            print("[i] load checkpoint from:", args["resume"], "for inference")
        elif "optimizer" in checkpoint and "lr_scheduler" in checkpoint and "epoch" in checkpoint:
            # optimizer.load_state_dict(checkpoint["optimizer"])
            # lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args["train"]["start_epoch"] = checkpoint["epoch"] + 1        
            print("[i] load checkpoint from:", args["resume"], "for train")
        else:
            print("[i] failed to load checkpoint from:", args["resume"])
            return        

    if args["infer"]:
        eval_infos = {
        "device": device,
        "is_local": is_local,
        "dim_feature": args["model"]["dim_feature"],        
        }        
        evaluate(model, criterion, postprocessors, data_loader_valid, eval_infos)
        return
    #                                                                                                               #
    #                                                                                                               #
    #################################################################################################################

    print("[i] start training ~")
    train_infos = {
        "iter" : 0,
        "epoch": -1,
        "device": device,
        "is_local": is_local,
        "clip_max_norm": args["train"]["clip_max_norm"],
    }
    valid_infos = {
        "epoch": -1,
        "device": device,
        "best_metric": -1,
        "is_local": is_local,
        "dim_feature": args["model"]["dim_feature"],        
    }
     
    for epoch in range(args["train"]["start_epoch"], args["train"]["epochs"] + 1):
        # if args["distributed"]:
        #     sampler_train.set_epoch(epoch)

        train_infos["epoch"] = epoch
        train_infos = train_one_epoch(
                model, criterion, postprocessors, data_loader_train, optimizer, train_infos, summary)
        lr_scheduler.step() 
            
        valid_infos["epoch"] = epoch
        valid_infos = valid_one_epoch(model, criterion, postprocessors, data_loader_valid, valid_infos, summary)   
        
        is_best = False
        if valid_infos["metric"] > valid_infos["best_metric"]:
            valid_infos["best_metric"] = valid_infos["metric"]
            is_best = True       
            
        save_state(out_dir, model, optimizer, lr_scheduler, epoch, is_best)

if __name__ == "__main__":
    main()
