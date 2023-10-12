# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
from typing import Iterable
from tqdm import tqdm

import torch
import numpy as np
import tensorboardX

import random 
from common.utils import AverageMeter, retr_accuracy, local_accuracy
from common.utils_summary import update_summary, plot_result

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module, postprocessors: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    train_infos: dict, summary: tensorboardX.SummaryWriter
                    ):
    
    model.train()
    # criterion.train()
    
    losses_meter = {}
    for k in criterion.losses: losses_meter[k] = AverageMeter()
    # losses_meter = AverageMeter()
    
    plot_imgs = {}
    
    iters = train_infos["iter"]
    
    sample_ind = random.choice(range(len(data_loader)))    
    description = "[i] Train {:>2}".format(train_infos["epoch"])
    for i, (img_grd, img_arl, targets) in \
        enumerate(tqdm(data_loader, desc=description, unit="batches")):
        
        bs = img_grd.size(0)
        img_grd = img_grd.to(train_infos["device"])
        img_arl = img_arl.to(train_infos["device"])
        
        outputs = model(im_grd=img_grd, im_arl=img_arl)  
        
        if train_infos["IS_POSE"]:
            targets = [{k: v.to(train_infos["device"]) for k, v in t.items()} for t in targets]
            results = postprocessors["bbox"](outputs, targets)
            if i == sample_ind: 
                p_imgs = plot_result(img_grd, img_arl, targets, results, th=0.1)
                plot_imgs.update(p_imgs)

        # losses, mean_p, mean_n = criterion(outputs["grd"], outputs["arl"])
        # losses_meter.update(losses.item(), bs)
        loss_dict = criterion(outputs, targets)
        losses = sum(loss_dict[k] for k in loss_dict.keys())        
        for k in loss_dict.keys(): losses_meter[k].update(loss_dict[k].item(), bs)
        
        # compute gradient and do SGD step        
        optimizer.zero_grad()
        losses.backward()
        # if train_infos["clip_max_norm"] > 0:
        #     torch.nn.utils.clip_grad_norm_(model.parameters(), train_infos["clip_max_norm"])
        if train_infos["optimizer"] != 'sam':
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
    for k in plot_imgs.keys(): imgs["image/" + k] = plot_imgs[k]
    
    stats = {}
    loss_total = 0
    for k in losses_meter.keys():
        stats["loss/" + k] =  losses_meter[k].avg
        loss_total += losses_meter[k].avg
    stats["loss/total"] = loss_total
    # stats["loss/total"] = losses_meter.avg
    
    update_summary(summary, imgs, stats, train_infos["epoch"], "train")
        
    for k in stats.keys():
        print("   ", k + ": {:.8f}".format(stats[k]), end = "\n")
    
    train_infos["iter"] = iters
    return train_infos

def valid_one_epoch(model: torch.nn.Module, 
                    criterion: torch.nn.Module,
                    postprocessors: torch.nn.Module,
                    loader_dict: dict, 
                    valid_infos: dict, 
                    summary: tensorboardX.SummaryWriter
                    ):    
    
    # retrieval validation
    imgs, stats = valid_retr(model, loader_dict["qry"], loader_dict["ref"], valid_infos)
    valid_infos["metric"] = stats["acc/retr_top1"]    
    
    if valid_infos["IS_POSE"]:  
        imgs2, stats2 = valid_local(model, criterion, postprocessors, loader_dict["val"], valid_infos)        
        imgs.update(imgs2)
        stats.update(stats2)
    
    update_summary(summary, imgs, stats, valid_infos["epoch"], "valid")
    
    print("[i] Valid {:>2}:".format(valid_infos["epoch"]), end = "\n")
    for k in stats.keys():
        print("   ", k + ": {:.8f}".format(stats[k]), end = "\n")

    return valid_infos

def valid_retr(model: torch.nn.Module, 
               qry_loader: Iterable, 
               ref_loader: Iterable, 
               valid_infos: dict):
    
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
        description = "[i] Valid qry"
        for i, (img_grd, idx_grd, labels) in enumerate(tqdm(qry_loader, desc=description, unit="batches")):
            
            img_grd = img_grd.to(valid_infos["device"])
            idx_grd = idx_grd.to(valid_infos["device"])
            labels = labels.to(valid_infos["device"])
            
            out_emb_grd = model_query(img_grd)
            qry_feat[idx_grd.cpu().numpy(), :] = out_emb_grd.detach().cpu().numpy()
            qry_label[idx_grd.cpu().numpy()] = labels.detach().cpu().numpy()
                        
            if i == 0: img_grd_ = img_grd[0, :, :, :]
            
        # reference features
        description = "[i] Valid ref"
        for i, (img_arl, idx_arl, _) in enumerate(tqdm(ref_loader, desc=description, unit="batches")):
            
            img_arl = img_arl.to(valid_infos["device"])            
            out_emb_arl = model_reference(img_arl)  # delta           
             
            ref_feat[idx_arl.cpu().numpy(), :] = out_emb_arl.detach().cpu().numpy()
            if i == 0: img_arl_ = img_arl[0, :, :, :]

        retr_acc = retr_accuracy(qry_feat, ref_feat, qry_label.astype(int))
        
    imgs = {
        # "image/ir/grd": img_grd_,
        # "image/ir/arl": img_arl_,
    }
    stats = {
        "acc/retr_top1": retr_acc[0],
        "acc/retr_top5": retr_acc[1],
        "acc/retr_top10": retr_acc[2],
        "acc/retr_top1pc": retr_acc[3],
    }
             
    return imgs, stats

def valid_local(model: torch.nn.Module, 
                criterion: torch.nn.Module,
                postprocessors: torch.nn.Module,
                data_loader: Iterable, 
                valid_infos: dict):
    
    model.eval()
    criterion.eval()
    
    losses_meter = {}
    for k in criterion.losses: losses_meter[k] = AverageMeter()
    
    acc1s, acc5s, denoms = 0, 0, 0
    trs_errs, rot_errs = [], []
    
    sample_ind = random.choice(range(len(data_loader)))    
    with torch.no_grad():
        description = "[i] Valid loc"
        for i, (img_grd, img_arl, targets) in enumerate(tqdm(data_loader, desc=description, unit="batches")):
            
            img_grd = img_grd.to(valid_infos["device"])
            img_arl = img_arl.to(valid_infos["device"])
            targets = [{k: v.to(valid_infos["device"]) for k, v in t.items()} for t in targets]
            
            outputs = model(im_grd=img_grd, im_arl=img_arl)
            results = postprocessors["bbox"](outputs, targets)

            loss_dict = criterion(outputs, targets)      
            for k in loss_dict.keys(): losses_meter[k].update(loss_dict[k].item(), img_grd.tensors.size(0))
            
            if i == sample_ind: 
                plot_imgs = plot_result(img_grd, img_arl, targets, results, 0.1)
                
            acc1, acc5, trs_err, rot_err = local_accuracy(targets, results)
            acc1s += acc1
            acc5s += acc5
            denoms += img_grd.tensors.size(0)
            trs_errs.extend(trs_err)
            rot_errs.extend(rot_err)
                
    imgs = {}
    for k in plot_imgs.keys(): imgs["image/" + k] = plot_imgs[k]
    
    stats = {}
    loss_total = 0
    for k in losses_meter.keys():
        stats["loss/" + k] =  losses_meter[k].avg
        loss_total += losses_meter[k].avg
    stats["loss/total"] = loss_total
    
    stats["acc/local_d1"] = (acc1s / denoms) * 100
    stats["acc/local_d5"] = (acc5s / denoms) * 100
    
    stats["acc/local_trs_mean"] = np.mean(trs_errs)
    stats["acc/local_trs_median"] = np.median(trs_errs)
    stats["acc/local_rot_mean"] = np.mean(rot_errs)
    stats["acc/local_rot_median"] = np.median(rot_errs)
             
    return imgs, stats

@torch.no_grad()
def evaluate(model: torch.nn.Module, 
             criterion: torch.nn.Module,
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
    
    img_grd_, img_arl_ = None, None
    # query features
    description = "[i] Eval qry"
    for i, (img_grd, idx_grd, labels) in enumerate(tqdm(loader_dict["qry"], desc=description, unit="batches")):
        
        img_grd = img_grd.to(eval_infos["device"])
        idx_grd = idx_grd.to(eval_infos["device"])
        labels = labels.to(eval_infos["device"])
        
        # compute output
        out_emb_grd, _, _, _ = model_query(img_grd)
        qry_feat[idx_grd.cpu().numpy(), :] = out_emb_grd.cpu().numpy()
        qry_label[idx_grd.cpu().numpy()] = labels.cpu().numpy()
                    
        if i == 0: img_grd_ = img_grd[0, :, :, :]
        
    # reference features
    description = "[i] Eval ref"
    for i, (img_arl, idx_arl, _) in enumerate(tqdm(loader_dict["ref"], desc=description, unit="batches")):
        
        img_arl = img_arl.to(eval_infos["device"])            
        out_emb_arl, _, _, _ = model_reference(img_arl)  # delta           
            
        ref_feat[idx_arl.cpu().numpy(), :] = out_emb_arl.detach().cpu().numpy()
        if i == 0: img_arl_ = img_arl[0, :, :, :]

    retr_acc = retr_accuracy(qry_feat, ref_feat, qry_label.astype(int))
        
    stats = {
        "acc/retr_top1": retr_acc[0],
        "acc/retr_top5": retr_acc[1],
        "acc/retr_top10": retr_acc[2],
        "acc/retr_top1pc": retr_acc[3],
    }   
    
    if eval_infos["IS_POSE"]:   
    
        model.eval()
        criterion.eval()
        
        losses_meter = {}
        for k in criterion.losses: losses_meter[k] = AverageMeter()
        
        acc1s, acc5s, denoms = 0, 0, 0
        description = "[i] Eval loc"
        for i, (img_grd, img_arl, targets) in enumerate(tqdm(loader_dict["val"], desc=description, unit="batches")):
            
            img_grd = img_grd.to(eval_infos["device"])
            img_arl = img_arl.to(eval_infos["device"])
            targets = [{k: v.to(eval_infos["device"]) for k, v in t.items()} for t in targets]
            
            outputs = model(im_grd=img_grd, im_arl=img_arl)
            results = postprocessors["bbox"](outputs, targets)

            loss_dict = criterion(outputs, targets)      
            for k in loss_dict.keys(): losses_meter[k].update(loss_dict[k].item(), img_grd.tensors.size(0))
                
            acc1, acc5, trs_mean, trs_median, rot_mean, rot_median = local_accuracy(targets, results)
            acc1s += acc1
            acc5s += acc5
            denoms += img_grd.tensors.size(0)
                    
        loss_total = 0
        for k in losses_meter.keys():
            stats["loss/" + k] =  losses_meter[k].avg
            loss_total += losses_meter[k].avg
        stats["loss/total"] = loss_total
        
        stats["acc/local_d1"] = (acc1s / denoms) * 100
        stats["acc/local_d5"] = (acc5s / denoms) * 100
    
    print("[i] Eval ", end = "\n")
    for k in stats.keys():
        print("   ", k + ": {:.8f}".format(stats[k]), end = "\n")

    return 