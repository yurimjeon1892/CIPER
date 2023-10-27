import numpy as np
import random

from PIL import Image, ImageDraw

def plot_result(results, targets, img_grd, img_arl):
    
    rand_ind = random.choice(range(img_grd.size(0)))       
    img_grd_ = img_grd[rand_ind, :, :, :].detach().cpu().numpy()
    img_arl_ = img_arl[rand_ind, :, :, :].detach().cpu().numpy()
    
    if "scores" in results[rand_ind].keys():
        
        arl_img_size = targets[rand_ind]["orig_size"].detach().cpu().numpy()
        meter_per_pixel = targets[rand_ind]["meter_per_pixel"][0].detach().cpu().numpy()
        
        tgt = targets[rand_ind]["boxes"][0].detach().cpu().numpy()   
        tgt = np.array([[tgt[0] * arl_img_size[0] * meter_per_pixel,
                         tgt[1] * arl_img_size[1] * meter_per_pixel,
                         np.arctan2(tgt[3], tgt[2])]])         
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
    return imgs 

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