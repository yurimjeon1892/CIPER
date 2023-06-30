import os
import torch
import numpy as np
import shutil
import matplotlib.pyplot as plt

import random

from PIL import Image, ImageDraw

def summary_image_draw(imgs):    
    outs = {}    
    for k in imgs.keys():
        if torch.is_tensor(imgs[k]): img_np = imgs[k].cpu().detach().numpy()
        else: img_np = imgs[k]        
        img_np = img_np.astype('float')
        img_np = (img_np - np.min(img_np)) / (np.max(img_np) - np.min(img_np)) * 255
        img_np = img_np.astype('uint8') 
        if img_np.shape[2] == 3:
            img_np = np.transpose(img_np, (2, 0, 1))
        outs[k] = img_np
    return outs
    
def update_summary(summary, imgs, stats, iter, mode):

    for k in stats.keys():
        summary.add_scalar(mode + "/" + k, stats[k], iter) 

    imgs = summary_image_draw(imgs) 
    for k in list(imgs.keys()):
        summary.add_image(mode + '/' + k, imgs[k], iter)    

    return

def minmax_color_img_from_img_numpy(img, cmap=plt.cm.plasma):
    """
    :param img: Input image (numpy array, H x W)
    :param cmap: plt color map
    :return img: minmax colored image (numpy array, H x W x 3)
    """
    img = (img - np.min(img)) / (np.max(img) - np.min(img)) 
    minmax_img = 255 * cmap(img)[:, :, :3]
    # minmax_img = minmax_img.astype('uint8')
    return minmax_img

def plot_estimation_result(img_gnd, img_arl, gts, outputs, score_th=0.5):
    
    rand_ind = random.choice(range(img_gnd.tensors.size(0)))                
    img_gnd_ = img_gnd.tensors[rand_ind, :, :, :].detach().cpu().numpy()
    img_arl_ = img_arl.tensors[rand_ind, :, :, :].detach().cpu().numpy()
    
    if "pred_logits" in outputs.keys():
        
        bs, n = outputs['pred_logits'].shape[:2]
        n_sqrt = int(np.sqrt(n))        
        
        arl_img_size = img_arl_.shape[-1]
        
        src_logits = outputs['pred_logits'][rand_ind, :, :].view((n_sqrt, n_sqrt, -1)).detach().cpu().numpy()
        src_boxes = outputs['pred_boxes'][rand_ind, :, :].view((n_sqrt, n_sqrt, -1)).detach().cpu().numpy()
                        
        target_classes = torch.cat([v["labels"] for v in gts]).view((bs, n, -1))
        target_boxes = torch.cat([v["boxes"] for v in gts]).view((bs, n, -1)) 
        target_classes = target_classes[rand_ind, :, :].view((n_sqrt, n_sqrt, -1)).float().detach().cpu().numpy()
        target_boxes = target_boxes[rand_ind, :, :].view((n_sqrt, n_sqrt, -1)).detach().cpu().numpy()
        
        pred_logit = minmax_color_img_from_img_numpy(src_logits[:, :, 0], plt.cm.gray)
        gt_label = minmax_color_img_from_img_numpy(target_classes[:, :, 0], plt.cm.gray)
        
        pred_mask = src_logits[:, :, 0] > score_th
        gt_mask = target_classes[:, :, 0] > 0.5
        
        src_boxes = src_boxes[pred_mask]
        target_boxes = target_boxes[gt_mask]
        
        img_arl_ = np.transpose(img_arl_, (1, 2, 0)).copy() 
        img_arl_ = (img_arl_ - np.min(img_arl_)) / (np.max(img_arl_) - np.min(img_arl_))  
        
        radius = 5   
        
        pred_bbox = Image.fromarray(np.uint8(np.array(img_arl_).copy()*255))
        gt_bbox = Image.fromarray(np.uint8(np.array(img_arl_).copy()*255))
        
        draw1 = ImageDraw.Draw(pred_bbox)
        for i in range(src_boxes.shape[0]):
            px, py = int(src_boxes[i, 0] * arl_img_size), int(src_boxes[i, 1] * arl_img_size)
            draw1.ellipse([(py - radius, px - radius), (py + radius, px + radius)], fill="blue")
            
        draw2 = ImageDraw.Draw(gt_bbox)
        for i in range(target_boxes.shape[0]):
            px, py = int(target_boxes[i, 0] * arl_img_size), int(target_boxes[i, 1] * arl_img_size)
            draw2.ellipse([(py - radius, px - radius), (py + radius, px + radius)], fill="red")        
            theta_rad = float(target_boxes[i, 2])
            draw2.line([(py, px), (py + 25 * np.sin(theta_rad), px + 25 * np.cos(theta_rad ))], fill="red", width=3)
            
        pred_bbox = np.array(pred_bbox)
        gt_bbox = np.array(gt_bbox)
        
        imgs = {
            "gnd": img_gnd_,
            # "arl": img_arl_,
            
            # "pred_logit": pred_logit,
            "gt_label": gt_label,
            
            # "pred_bbox": pred_bbox,
            "gt_bbox": gt_bbox,
        }
    else:        
        imgs = {
            "gnd": img_gnd_,
            "arl": img_arl_,
        }
    return imgs