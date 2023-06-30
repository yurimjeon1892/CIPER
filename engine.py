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
from common.utils import AverageMeter, accuracy
from common.utils_summary import update_summary, plot_estimation_result

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    train_infos: dict, summary: tensorboardX.SummaryWriter
                    ):
    
    model.train()
    criterion.train()
    
    losses_meter = {}
    for k in criterion.losses: losses_meter[k] = AverageMeter()
    
    iters = train_infos["iter"]
    
    sample_ind = random.choice(range(len(data_loader)))    
    description = '[i] Train {:>2}'.format(train_infos["epoch"])
    for i, (img_grnd, img_arl, gts) in \
        enumerate(tqdm(data_loader, desc=description, unit="batches")):
        
        bs = img_grnd.tensors.size(0)
        
        img_grnd = img_grnd.to(train_infos["device"])
        img_arl = img_arl.to(train_infos["device"])

        outputs = model(im_grnd=img_grnd, im_arl=img_arl)

        loss_dict = criterion(outputs, gts)
        losses = sum(loss_dict[k] for k in loss_dict.keys())        
        for k in loss_dict.keys(): losses_meter[k].update(loss_dict[k].item(), bs)
        
        if i == sample_ind: 
            plot_imgs = plot_estimation_result(img_grnd, img_arl, gts, outputs)
        
        # compute gradient and do SGD step        
        optimizer.zero_grad()
        losses.backward()
        if train_infos["clip_max_norm"] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_infos["clip_max_norm"])
        optimizer.step()
                        
        iters += bs
        
    imgs = {}
    for k in plot_imgs.keys(): imgs["image/train/" + k] = plot_imgs[k]
    
    stats = {}
    loss_total = 0
    for k in losses_meter.keys():
        stats["loss/" + k] =  losses_meter[k].avg
        loss_total += losses_meter[k].avg
    stats["loss/total"] = loss_total
    update_summary(summary, imgs, stats, train_infos["epoch"], "train")
        
    for k in stats.keys():
        print("   ", k + ": {:.4f}".format(stats[k]), end = '\n')
    
    train_infos["iter"] = iters
    return train_infos

def valid_one_epoch(model: torch.nn.Module, 
                    criterion: torch.nn.Module,
                    loader_dict: dict, 
                    valid_infos: dict, 
                    summary: tensorboardX.SummaryWriter
                    ):    
    
    # retrieval validation
    imgs, stats = valid_retrieval(model, loader_dict["qry"], loader_dict["ref"], valid_infos)
    valid_infos["metric"] = stats["acc/retrieval/top1"]    
    
    if valid_infos["task"] == "RUNNGUN":  
        imgs2, stats2 = valid_estimation(model, criterion, loader_dict["val"], valid_infos)        
        imgs.update(imgs2)
        stats.update(stats2)
        # valid_infos["metric"] = stats["acc/estimation/top1"]  
    
    update_summary(summary, imgs, stats, valid_infos["epoch"], "valid")
    
    print('[i] Valid {:>2}:'.format(valid_infos["epoch"]), end = '\n')
    for k in stats.keys():
        print("   ", k + ": {:.4f}".format(stats[k]), end = '\n')

    return valid_infos

def valid_retrieval(model: torch.nn.Module, 
                    qry_loader: Iterable, 
                    ref_loader: Iterable, 
                    valid_infos: dict):
    
    model_query = model.query_net
    model_reference = model.reference_net
    
    model_query.eval()
    model_reference.eval()

    qry_feat = np.zeros([len(qry_loader.dataset), valid_infos["hidden_dim"]])
    qry_label = np.zeros([len(qry_loader.dataset)])
    ref_feat = np.zeros([len(ref_loader.dataset), valid_infos["hidden_dim"]])
    
    img_grnd_, img_arl_ = None, None
    with torch.no_grad():
        # query features
        for i, (img_grnd, idx_grnd, labels) in enumerate(qry_loader):
            
            img_grnd = img_grnd.to(valid_infos["device"])
            idx_grnd = idx_grnd.to(valid_infos["device"])
            labels = labels.to(valid_infos["device"])
            
            # compute output
            out_emb_grnd, _ = model_query(img_grnd)

            qry_feat[idx_grnd.cpu().numpy(), :] = out_emb_grnd.cpu().numpy()
            qry_label[idx_grnd.cpu().numpy()] = labels.cpu().numpy()
                        
            if i == 0: img_grnd_ = img_grnd[0, :, :, :]
            
        # reference features
        for i, (img_arl, idx_arl, _) in enumerate(ref_loader):
            
            img_arl = img_arl.to(valid_infos["device"])            
            out_emb_arl, _ = model_reference(img_arl)  # delta            
            ref_feat[idx_arl.cpu().numpy(), :] = out_emb_arl.detach().cpu().numpy()
            if i == 0: img_arl_ = img_arl[0, :, :, :]

        [top1, top5] = accuracy(qry_feat, ref_feat, qry_label.astype(int))
        
    imgs = {
        # "image/retrieval/grnd": img_grnd_,
        # "image/retrieval/arl": img_arl_,
    }
    stats = {
        "acc/retrieval/top1": top1,
        "acc/retrieval/top5": top5,
    }
             
    return imgs, stats

def valid_estimation(model: torch.nn.Module, 
                    criterion: torch.nn.Module,
                    data_loader: Iterable, 
                    valid_infos: dict):
    
    model.eval()
    criterion.eval()
    
    losses_meter = {}
    for k in criterion.losses: losses_meter[k] = AverageMeter()
    
    sample_ind = random.choice(range(len(data_loader)))    
    with torch.no_grad():
        for i, (img_grnd, img_arl, gts) in enumerate(data_loader):
            
            img_grnd = img_grnd.to(valid_infos["device"])
            img_arl = img_arl.to(valid_infos["device"])
            
            outputs = model(im_grnd=img_grnd, im_arl=img_arl)
            
            bs = img_grnd.tensors.size(0)

            loss_dict = criterion(outputs, gts)      
            for k in loss_dict.keys(): losses_meter[k].update(loss_dict[k].item(), bs)
            
            if i == sample_ind: 
                plot_imgs = plot_estimation_result(img_grnd, img_arl, gts, outputs)
                
    imgs = {}
    for k in plot_imgs.keys(): imgs["image/estimation/" + k] = plot_imgs[k]
    
    stats = {}
    loss_total = 0
    for k in losses_meter.keys():
        stats["loss/estimation/" + k] =  losses_meter[k].avg
        loss_total += losses_meter[k].avg
    stats["loss/estimation/total"] = loss_total
             
    return imgs, stats