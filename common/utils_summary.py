import os
import torch
import numpy as np
import shutil
import matplotlib.pyplot as plt

import random

from PIL import Image, ImageDraw

def update_summary(summary, imgs, stats, iter, mode):

    for k in stats.keys():
        summary.add_scalar(mode + "/" + k, stats[k], iter) 

    imgs = summary_image_draw(imgs) 
    for k in list(imgs.keys()):
        summary.add_image(mode + '/' + k, imgs[k], iter)    

    return

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

def plot_result(img_gnd, img_arl, targets, results, th):
    
    rand_ind = random.choice(range(img_gnd.tensors.size(0)))       
    img_gnd_ = img_gnd.tensors[rand_ind, :, :, :].detach().cpu().numpy()
    img_arl_ = img_arl.tensors[rand_ind, :, :, :].detach().cpu().numpy()
    
    if "scores" in results[rand_ind].keys():
        
        arl_img_size = img_arl_.shape[1:]        
        
        arl_zoom_ratio = targets[rand_ind]["arl_zoom_ratio"][0].detach().cpu().numpy()
        meter_per_pixel = targets[rand_ind]["meter_per_pixel"][0].detach().cpu().numpy()
        
        src_scores = results[rand_ind]["scores"].detach().cpu().numpy()
        src_boxes = results[rand_ind]['boxes'].detach().cpu().numpy()        
        src_boxes[:, :2] /= (arl_zoom_ratio * meter_per_pixel) 
        src_bbox_img = plot_pred_dot(img_arl_, src_boxes, src_scores > th, arl_img_size, "cyan")
        
        target_boxes = torch.cat([v["boxes"] for v in targets])
        target_boxes = target_boxes[rand_ind].detach().cpu().numpy()                    
        target_bbox_img = plot_gt_dot(img_arl_, target_boxes, arl_img_size, "red")
        
        img_bbox = np.concatenate([src_bbox_img, target_bbox_img], 1)
        
        imgs = {
            "1_gnd": img_gnd_,
            "2_bbox": img_bbox,
        }
    else:        
        imgs = {
            "1_gnd": img_gnd_,
            "1_arl": img_arl_,
        }
    return imgs

def draw_pin(draw, x, y, theta, img_size, color, radius):
    px, py = int(x + img_size[0] / 2), int(y + img_size[1] / 2)
    draw.ellipse([(py - radius, px - radius), (py + radius, px + radius)], fill=color)     
    draw.line([(py, px), (py + 25 * np.sin(theta), px + 25 * np.cos(theta ))], fill=color, width=3)
    return draw   
    

def plot_pred_dot(img_np, boxes, is_valid, img_size, color, radius=5):    
    
    img_np = np.transpose(img_np, (1, 2, 0)).copy() 
    img_np = (img_np - np.min(img_np)) / (np.max(img_np) - np.min(img_np))  
    img = Image.fromarray(np.uint8(np.array(img_np).copy() * 255))
    draw = ImageDraw.Draw(img)
    
    for i in range(boxes.shape[0]):
        if is_valid[i] == False: continue
        px, py, theta = boxes[i, 0], boxes[i, 1], boxes[i, 2]
        draw_pin(draw, px, py, theta, img_size, color, radius)
        
    return np.array(img)

def plot_gt_dot(img_np, box, img_size, color, radius=5):    
    
    img_np = np.transpose(img_np, (1, 2, 0)).copy() 
    img_np = (img_np - np.min(img_np)) / (np.max(img_np) - np.min(img_np))  
    img = Image.fromarray(np.uint8(np.array(img_np).copy() * 255))
    draw = ImageDraw.Draw(img)
        
    px, py = box[0] * img_size[1], box[1] * img_size[0]
    theta = np.arctan2(box[3], box[2])
    
    draw_pin(draw, px, py, theta, img_size, color, radius)
    
    return np.array(img)