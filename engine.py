"""
Train and eval functions used in main.py
"""
from typing import Iterable
from tqdm import tqdm

import torch
import numpy as np

import random 
from common.utils import AverageMeter, retr_accuracy, pose_accuracy
from common.utils_plot import plot_result
import wandb

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module, postprocessors: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer, train_infos: dict, 
                    ):
    
    model.train()
    # criterion.train()
    
    losses_meter = {}
    for k in criterion.losses: losses_meter[k] = AverageMeter()
    
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
        
        if not train_infos["retr_only"]:
            targets = [{k: targets[k][b].to(train_infos["device"]) for k in targets.keys()} for b in range(bs) ]
            results = postprocessors["bbox"](outputs, targets)
            if i == sample_ind: 
                p_imgs = plot_result(results, targets, img_grd, img_arl)
                plot_imgs.update(p_imgs)

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
    for k in plot_imgs.keys(): 
        if plot_imgs[k].shape[0] == 3:
            plot_imgs[k] = np.transpose(plot_imgs[k], (1, 2, 0))
        imgs["train_image/" + k] = wandb.Image(plot_imgs[k])
    wandb.log(imgs, step=train_infos["epoch"])
    
    stats = {}
    loss_total = 0
    for k in losses_meter.keys():
        stats["train_loss/" + k] =  losses_meter[k].avg
        loss_total += losses_meter[k].avg
    stats["train_loss/total"] = loss_total
    wandb.log(stats, step=train_infos["epoch"])
            
    for k in stats.keys():
        print("   ", k + ": {:.8f}".format(stats[k]), end = "\n")
    
    train_infos["iter"] = iters
    return train_infos

def valid_one_epoch(model: torch.nn.Module, 
                    criterion: torch.nn.Module,
                    postprocessors: torch.nn.Module,
                    loader_dict: dict, 
                    valid_infos: dict, 
                    ):    
    
    # retrieval validation
    imgs, stats = valid_retr(model, loader_dict["qry"], loader_dict["ref"], valid_infos)
    valid_infos["metric"] = stats["valid_acc/retr_top1"] 
    wandb.log(stats, step=valid_infos["epoch"])
    
    if valid_infos["data_name"] == "kitti":
        imgs_val2_retr, stats_val2_retr = valid_retr(model, loader_dict["qry2"], loader_dict["ref2"], valid_infos)
        stats_val2_retrn = {}
        for k in stats_val2_retr.keys():
            nk = k.replace("valid", "valid2")
            stats_val2_retrn[nk] = stats_val2_retr[k]
        wandb.log(stats_val2_retrn, step=valid_infos["epoch"]); stats.update(stats_val2_retrn)
    
    if not valid_infos["retr_only"]:  
        imgs_val_pose, stats_val_pose = valid_pose(model, criterion, postprocessors, loader_dict["val"], valid_infos)        
        valid_infos["metric"] = stats_val_pose["valid_acc/pose_trs_d1"] 
        wandb.log(imgs_val_pose, step=valid_infos["epoch"]) 
        wandb.log(stats_val_pose, step=valid_infos["epoch"]) 
        stats.update(stats_val_pose)
        
        if valid_infos["data_name"] == "kitti":
            imgs_val2_pose, stats_val2_pose = valid_pose(model, criterion, postprocessors, loader_dict["val2"], valid_infos)  
            
            imgs_val2_posen, stats_val2_posen = {}, {}
            for k in imgs_val2_pose.keys():
                nk = k.replace("valid", "valid2")
                imgs_val2_posen[nk] = imgs_val2_pose[k]

            for k in stats_val2_pose.keys():
                nk = k.replace("valid", "valid2")
                stats_val2_posen[nk] = stats_val2_pose[k]
            
            wandb.log(imgs_val2_posen, step=valid_infos["epoch"]) 
            wandb.log(stats_val2_posen, step=valid_infos["epoch"]) 
            stats.update(stats_val2_posen)
        
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
            
            out_emb_grd, _, _ = model_query(img_grd)
            qry_feat[idx_grd.cpu().numpy(), :] = out_emb_grd.detach().cpu().numpy()
            qry_label[idx_grd.cpu().numpy()] = labels.detach().cpu().numpy()
                        
            if i == 0: img_grd_ = img_grd[0, :, :, :]
            
        # reference features
        description = "[i] Valid ref"
        for i, (img_arl, idx_arl, _) in enumerate(tqdm(ref_loader, desc=description, unit="batches")):
            
            img_arl = img_arl.to(valid_infos["device"])            
            out_emb_arl, _ = model_reference(img_arl)  # delta           
             
            ref_feat[idx_arl.cpu().numpy(), :] = out_emb_arl.detach().cpu().numpy()
            if i == 0: img_arl_ = img_arl[0, :, :, :]

        retr_acc = retr_accuracy(qry_feat, ref_feat, qry_label.astype(int))
        
    imgs = {
        # "image/ir/grd": img_grd_,
        # "image/ir/arl": img_arl_,
    }
    stats = {
        "valid_acc/retr_top1": retr_acc[0],
        "valid_acc/retr_top5": retr_acc[1],
        "valid_acc/retr_top10": retr_acc[2],
        "valid_acc/retr_top1pc": retr_acc[3],
    }
             
    return imgs, stats

def valid_pose(model: torch.nn.Module, 
               criterion: torch.nn.Module,
               postprocessors: torch.nn.Module,
               data_loader: Iterable, 
               valid_infos: dict):
    
    model.eval()
    criterion.eval()
    
    losses_meter = {}
    for k in criterion.losses: losses_meter[k] = AverageMeter()
    
    trs_errs, rot_errs = [], []    
    # sample_ind = random.choice(range(len(data_loader)))  
    sample_ind = 0  
    with torch.no_grad():
        description = "[i] Valid pose"
        for i, (img_grd, img_arl, targets) in enumerate(tqdm(data_loader, desc=description, unit="batches")):
            
            img_grd = img_grd.to(valid_infos["device"])
            img_arl = img_arl.to(valid_infos["device"])
            targets = [ {k: targets[k][b].to(valid_infos["device"]) for k in targets.keys()} for b in range(img_grd.size(0)) ]
            
            outputs = model(im_grd=img_grd, im_arl=img_arl)
            results = postprocessors["bbox"](outputs, targets)

            loss_dict = criterion(outputs, targets)      
            for k in loss_dict.keys(): losses_meter[k].update(loss_dict[k].item(), img_grd.size(0))
            
            if i == sample_ind: 
                plot_imgs = plot_result(results, targets, img_grd, img_arl)
                
            trs_err, rot_err = pose_accuracy(results, targets)
            trs_errs.extend(trs_err)
            rot_errs.extend(rot_err)
                
    imgs = {}
    for k in plot_imgs.keys(): 
        if plot_imgs[k].shape[0] == 3:
            plot_imgs[k] = np.transpose(plot_imgs[k], (1, 2, 0))
        imgs["valid_image/" + k] = wandb.Image(plot_imgs[k])
    
    stats = {}
    loss_total = 0
    for k in losses_meter.keys():
        if "retr" in k : continue
        stats["valid_loss/" + k] =  losses_meter[k].avg
        loss_total += losses_meter[k].avg
    stats["valid_loss/total"] = loss_total
    
    stats["valid_acc/pose_trs_d1"] = np.sum((trs_err < 1)) / trs_err.shape[0] * 100
    stats["valid_acc/pose_trs_d5"] = np.sum((trs_err < 5)) / trs_err.shape[0] * 100
    
    stats["valid_acc/pose_rot_d1"] = np.sum((rot_err < 1)) / rot_err.shape[0] * 100
    stats["valid_acc/pose_rot_d5"] = np.sum((rot_err < 5)) / rot_err.shape[0] * 100
    
    stats["valid_err/pose_trs_mean(m)"] = np.mean(trs_errs)
    stats["valid_err/pose_trs_median(m)"] = np.median(trs_errs)
    stats["valid_err/pose_rot_mean(deg)"] = np.mean(rot_errs)
    stats["valid_err/pose_rot_median(deg)"] = np.median(rot_errs)
             
    return imgs, stats

@torch.no_grad()
def evaluate(model: torch.nn.Module, 
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
    
    description = "[i] Eval qry"
    for i, (img_grd, idx_grd, labels) in enumerate(tqdm(loader_dict["qry"], desc=description, unit="batches")):
        
        img_grd = img_grd.to(eval_infos["device"])
        idx_grd = idx_grd.to(eval_infos["device"])
        labels = labels.to(eval_infos["device"])
        
        out_emb_grd, _ = model_query(img_grd)
        qry_feat[idx_grd.cpu().numpy(), :] = out_emb_grd.cpu().numpy()
        qry_label[idx_grd.cpu().numpy()] = labels.cpu().numpy()
    
    description = "[i] Eval ref"
    for i, (img_arl, idx_arl, _) in enumerate(tqdm(loader_dict["ref"], desc=description, unit="batches")):
        img_arl = img_arl.to(eval_infos["device"])            
        out_emb_arl, _ = model_reference(img_arl)  # delta           
        
        ref_feat[idx_arl.cpu().numpy(), :] = out_emb_arl.detach().cpu().numpy()
    
    retr_acc = retr_accuracy(qry_feat, ref_feat, qry_label.astype(int))
        
    stats = {
        "acc/retr_top1": retr_acc[0],
        "acc/retr_top5": retr_acc[1],
        "acc/retr_top10": retr_acc[2],
        "acc/retr_top1pc": retr_acc[3],
    }   
    
    if eval_infos["retr_only"] == False:   
    
        model.eval()
        trs_errs, rot_errs = [], []    
                
        description = "[i] Eval pose"
        for i, (img_grd, img_arl, targets) in enumerate(tqdm(loader_dict["val"], desc=description, unit="batches")):
            
            img_grd = img_grd.to(eval_infos["device"])
            img_arl = img_arl.to(eval_infos["device"])
            targets = [ {k: targets[k][b].to(eval_infos["device"]) for k in targets.keys()} for b in range(img_grd.size(0))]
            
            outputs = model(im_grd=img_grd, im_arl=img_arl)
            results = postprocessors["bbox"](outputs, targets)
                
            trs_err, rot_err = pose_accuracy(results, targets)
            trs_errs.extend(trs_err)
            rot_errs.extend(rot_err)
            
        stats = {}
                    
        stats["acc/pose_trs_d1"] = np.sum((trs_err < 1)) / trs_err.shape[0] * 100
        stats["acc/pose_trs_d5"] = np.sum((trs_err < 5)) / trs_err.shape[0] * 100
        
        stats["acc/pose_rot_d1"] = np.sum((rot_err < 1)) / rot_err.shape[0] * 100
        stats["acc/pose_rot_d5"] = np.sum((rot_err < 5)) / rot_err.shape[0] * 100
        
        stats["err/pose_trs_mean"] = np.mean(trs_errs)
        stats["err/pose_trs_median"] = np.median(trs_errs)
        stats["err/pose_rot_mean"] = np.mean(rot_errs)
        stats["err/pose_rot_median"] = np.median(rot_errs)
    
    print("[i] Eval ", end = "\n")
    for k in stats.keys():
        print("   ", k + ": {:.8f}".format(stats[k]), end = "\n")

    return 
