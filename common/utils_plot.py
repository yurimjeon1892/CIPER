import numpy as np
import random, os

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


def plot_result(results, targets, img_grd, img_arl):
    rand_ind = 0
    img_grd_ = img_grd[rand_ind, :, :, :].detach().cpu().numpy()
    img_arl_ = img_arl[rand_ind, :, :, :].detach().cpu().numpy()

    if "scores" in results[rand_ind].keys():
        arl_img_size = targets[rand_ind]["orig_size"].detach().cpu().numpy()
        meter_per_pixel = targets[rand_ind]["meter_per_pixel"][0].detach().cpu().numpy()

        tgt = targets[rand_ind]["boxes"][0].detach().cpu().numpy()
        yaw = np.arctan2(tgt[3], tgt[2])
        tgt = np.array(
            [
                [
                    tgt[0] * arl_img_size[0] * meter_per_pixel,
                    tgt[1] * arl_img_size[1] * meter_per_pixel,
                    yaw,
                ]
            ]
        )
        target_img = draw_3dof_pin(
            img_arl_, tgt, arl_img_size, meter_per_pixel, "orange"
        )

        scores = results[rand_ind]["scores"].detach().cpu().numpy()
        shifts = results[rand_ind]["boxes"].detach().cpu().numpy()
        shifts_max = shifts[np.argmax(scores), :]
        shifts_max = np.array([[shifts_max[0], shifts_max[1], shifts_max[2]]])
        pred_img = draw_3dof_pin(
            img_arl_, shifts, arl_img_size, meter_per_pixel, "blue"
        )
        pred_img = draw_3dof_pin(
            pred_img, shifts_max, arl_img_size, meter_per_pixel, "cyan"
        )

        print("pred: ", shifts_max.astype(float))
        print("target: ", tgt.astype(float))

        img_bbox = np.concatenate([target_img, pred_img], 1)

        imgs = {
            "1_grd": img_grd_,
            "2_bbox": img_bbox,
        }
    else:
        imgs = {
            "1_grd": img_grd_,
            "1_arl": img_arl_,
        }

    if "rng_mask" in results[rand_ind].keys():
        rng_mask = results[rand_ind]["rng_mask"].detach().cpu().numpy()
        bev_mask = results[rand_ind]["bev_mask"].detach().cpu().numpy()

        rng_mask = np.tile(rng_mask[0], (32, 1))
        img_rng_mask = draw_minmax_color_img(rng_mask, cmap=plt.cm.plasma)

        n = int(bev_mask.shape[0] ** 0.5)
        bev_mask = np.reshape(bev_mask[:, 0], (n, n))
        img_bev_mask = draw_minmax_color_img(bev_mask, cmap=plt.cm.plasma)

        imgs["3_rng_mask"] = img_rng_mask
        imgs["3_bev_mask"] = img_bev_mask

    return imgs


def plot_intermediate(intermediate):
    rand_ind = 0

    target_mask = intermediate["target_mask"][rand_ind].detach().cpu().numpy()
    target_mask = np.tile(target_mask[0], (32, 1))
    img_target_mask = draw_minmax_color_img(target_mask, cmap=plt.cm.plasma)

    img = {"target_mask": img_target_mask}
    return img


def plot_data(img_grd, img_arl, targets, save_root, idx):
    img_grd = img_grd.detach().cpu().numpy()
    img_arl = img_arl.detach().cpu().numpy()

    for i in range(img_grd.shape[0]):
        arl_img_size = targets[i]["orig_size"].detach().cpu().numpy()
        meter_per_pixel = targets[i]["meter_per_pixel"][0].detach().cpu().numpy()
        tgt = targets[i]["boxes"][0].detach().cpu().numpy()
        yaw = np.arctan2(tgt[3], tgt[2])
        tgt = np.array(
            [
                [
                    tgt[0] * arl_img_size[0] * meter_per_pixel,
                    tgt[1] * arl_img_size[1] * meter_per_pixel,
                    yaw,
                ]
            ]
        )
        target_img = draw_3dof_pin(
            img_arl[i], tgt, arl_img_size, meter_per_pixel, "orange"
        )

        im = Image.fromarray(target_img)
        im = im.save(
            os.path.join(
                save_root, str(idx).zfill(4) + "_" + str(i).zfill(2) + "_arl.png"
            )
        )

        img_grd_ = img_grd[i]

        if img_grd_.shape[0] == 3:
            img_grd_ = np.transpose(img_grd_, (1, 2, 0)).copy()
        im2 = Image.fromarray(np.uint8(np.array(img_grd_).copy() * 255))
        im2 = im2.save(
            os.path.join(
                save_root, str(idx).zfill(4) + "_" + str(i).zfill(2) + "_grd.png"
            )
        )

    return


def draw_3dof_pin(img_np, boxes, img_size, meter_per_pixel, color, radius=5):
    if img_np.shape[0] == 3:
        img_np = np.transpose(img_np, (1, 2, 0)).copy()
    else:
        img_np = img_np.copy()
    img_np = (img_np - np.min(img_np)) / (np.max(img_np) - np.min(img_np))
    img = Image.fromarray(np.uint8(np.array(img_np).copy() * 255))

    if boxes.shape[0] == 0:
        return np.array(img)

    # boxes = np.nan_to_num(boxes)
    draw = ImageDraw.Draw(img)
    for i in range(boxes.shape[0]):
        px, py, theta = (
            boxes[i, 0] / meter_per_pixel,
            boxes[i, 1] / meter_per_pixel,
            boxes[i, 2],
        )
        px, py = int(px + img_size[0] / 2), int(py + img_size[1] / 2)
        draw.ellipse(
            [(py - radius, px - radius), (py + radius, px + radius)], fill=color
        )
        draw.line(
            [(py, px), (py + 25 * np.sin(theta), px + 25 * np.cos(theta))],
            fill=color,
            width=3,
        )
    return np.array(img)


def draw_minmax_color_img(img, cmap):
    """
    :param img: Input image (numpy array, H x W)
    :param cmap: plt color map
    :return img: minmax colored image (numpy array, H x W x 3)
    """
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    minmax_img = 255 * cmap(img)[:, :, :3]
    minmax_img = minmax_img.astype("uint8")
    return minmax_img
