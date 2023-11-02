import numpy as np
import random

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

def plot_result(results, targets, img_grd, img_arl):
    
    # rand_ind = random.choice(range(img_grd.size(0)))   
    rand_ind = 0    
    img_grd_ = img_grd[rand_ind, :, :, :].detach().cpu().numpy()
    img_arl_ = img_arl[rand_ind, :, :, :].detach().cpu().numpy()
    
    if "scores" in results[rand_ind].keys():
        
        arl_img_size = targets[rand_ind]["orig_size"].detach().cpu().numpy()
        meter_per_pixel = targets[rand_ind]["meter_per_pixel"][0].detach().cpu().numpy()
        
        tgt = targets[rand_ind]["boxes"][0].detach().cpu().numpy()   
        yaw = np.arctan2(tgt[3], tgt[2])
        tgt = np.array([[tgt[0] * arl_img_size[0] * meter_per_pixel,
                         tgt[1] * arl_img_size[1] * meter_per_pixel,
                         yaw]])         
        target_img = draw_3dof_pin(img_arl_, tgt, arl_img_size, meter_per_pixel, "orange")
        
        scores = results[rand_ind]["scores"].detach().cpu().numpy()
        shifts = results[rand_ind]['boxes'].detach().cpu().numpy()    
        shifts_max = shifts[np.argmax(scores), :]
        shifts_max = np.array([[shifts_max[0], shifts_max[1], shifts_max[2]]])
        pred_img = draw_3dof_pin(img_arl_, shifts, arl_img_size, meter_per_pixel, "blue")
        pred_img = draw_3dof_pin(pred_img, shifts_max, arl_img_size, meter_per_pixel, "cyan")
        
        print("pred: ", shifts_max.astype(float))
        print("target: ", tgt.astype(float))
        
        img_bbox = np.concatenate([target_img, pred_img], 1)
        
        imgs = {
            "1_gnd": img_grd_,
            "2_bbox": img_bbox,
        }
    else:        
        imgs = {
            "1_gnd": img_grd_,
            "1_arl": img_arl_,
        }
        
    if "attn1" in results[rand_ind].keys():
        
        ray_attn = results[rand_ind]["attn1"].detach().cpu().numpy()
        bev_attn = results[rand_ind]["attn2"].detach().cpu().numpy()
        
        ray_attn = np.tile(ray_attn[0], (32, 1))
        img_ray_attn = draw_minmax_color_img(ray_attn, cmap=plt.cm.plasma)
        
        n = int(bev_attn.shape[0] ** 0.5)
        bev_attn = np.reshape(bev_attn[:, 0], (n, n))
        img_bev_attn = draw_minmax_color_img(bev_attn, cmap=plt.cm.plasma)
        
        imgs["3_ray_attn"] = img_ray_attn
        imgs["3_bev_attn"] = img_bev_attn
        
    return imgs 

def plot_criterion_save(criterion_save):
    
    rand_ind = 0 
    target_attn = criterion_save["target_attn"][rand_ind].detach().cpu().numpy() 
    
    target_attn = np.tile(target_attn[0], (32, 1))
    img_target_attn = draw_minmax_color_img(target_attn, cmap=plt.cm.plasma)
        
    img = {
        "target_attn": img_target_attn
    }
    return img

def draw_3dof_pin(img_np, boxes, img_size, meter_per_pixel, color, radius=5):    
    
    if img_np.shape[0] == 3: img_np = np.transpose(img_np, (1, 2, 0)).copy() 
    else: img_np = img_np.copy() 
    img_np = (img_np - np.min(img_np)) / (np.max(img_np) - np.min(img_np))  
    img = Image.fromarray(np.uint8(np.array(img_np).copy() * 255))
    
    if boxes.shape[0] == 0: return np.array(img)
    
    draw = ImageDraw.Draw(img)    
    for i in range(boxes.shape[0]):
        px, py, theta = boxes[i, 0] / meter_per_pixel , boxes[i, 1] / meter_per_pixel, boxes[i, 2]
        px, py = int(px + img_size[0] / 2), int(py + img_size[1] / 2)
        draw.ellipse([(py - radius, px - radius), (py + radius, px + radius)], fill=color)     
        draw.line([(py, px), (py + 25 * np.sin(theta), px + 25 * np.cos(theta ))], fill=color, width=3)      
    return np.array(img)

def draw_minmax_color_img(img, cmap):
    """
    :param img: Input image (numpy array, H x W)
    :param cmap: plt color map
    :return img: minmax colored image (numpy array, H x W x 3)
    """
    img = (img - np.min(img)) / (np.max(img) - np.min(img))  
    minmax_img = 255 * cmap(img)[:, :, :3]
    minmax_img = minmax_img.astype('uint8')
    return minmax_img