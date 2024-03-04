import numpy as np

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


def plot_result(preds, gts, img_grd, img_arl):
    rand_ind = 0
    img_grd_ = img_grd[rand_ind, :, :, :].detach().cpu().numpy()
    img_arl_ = img_arl[rand_ind, :, :, :].detach().cpu().numpy()
    pred = preds[rand_ind]
    gt = gts[rand_ind]

    img_arl_ = draw_3dof_pin(img_arl_, gt[0], gt[1], gt[2], "orange")
    img_arl_ = draw_3dof_pin(img_arl_, pred[0], pred[1], pred[2], "cyan")

    imgs = {
        "1_grd": img_grd_,
        "1_arl": img_arl_,
    }

    return imgs


def draw_3dof_pin(img, px, py, theta, color, radius=5):
    if img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0)).copy()
    else:
        img = img.copy()
    img_size = img.shape[:2]

    px, py = int(px + img_size[0] / 2), int(py + img_size[1] / 2)
    theta = np.deg2rad(theta)

    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    img = Image.fromarray(np.uint8(np.array(img).copy() * 255))

    draw = ImageDraw.Draw(img)
    draw.ellipse([(px - radius, py - radius), (px + radius, py + radius)], fill=color)
    draw.line(
        [(px, py), (px + 25 * np.sin(theta), py + 25 * np.cos(theta))],
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
