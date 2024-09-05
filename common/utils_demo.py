import os
import torch
from torch import nn

from common.utils_loader import input_transform
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

import numpy as np

import os
import plotly.express as px
import pandas as pd
import datetime


def load_query(args):
    transform_query = input_transform(size=args["grd_img_size"])
    grd_img = Image.open(args["qry_path"], "r")
    grd_img = grd_img.convert("RGB")
    grd_img = transform_query(grd_img)
    return grd_img.unsqueeze(0)


def load_candidates(args, pred_meta_topk):
    transform_ref = input_transform(size=args["arl_img_size"])
    candidates = []
    for i in range(len(pred_meta_topk)):
        file_name = pred_meta_topk[i]["file_name"]
        arl_img = Image.open(os.path.join(args["data_root"], file_name), "r")
        arl_img = arl_img.convert("RGB")
        arl_img = transform_ref(arl_img)
        candidates.append(arl_img.unsqueeze(0))
    return candidates


def run_geo_localization(args, pred_meta_topk, pred_pose):

    r_earth = 6371000.0

    pred_locs_1 = []
    for i in range(len(pred_meta_topk)):
        lat = pred_meta_topk[i]["lat"].detach().cpu().numpy()
        lon = pred_meta_topk[i]["lon"].detach().cpu().numpy()
        yaw = pred_meta_topk[i]["yaw"].detach().cpu().numpy()
        pred_locs_1.append([lat, lon, yaw])

    # # convert x, y, theta to lat, lon, yaw
    pred_locs_2 = []
    for i in range(len(pred_pose)):
        dlat = (
            pred_pose[i][0][0][0]
            * pred_meta_topk[i]["meter_per_pixel"].detach().cpu().numpy()
            * args["arl_img_size"][0]
        )
        dlon = (
            pred_pose[i][0][0][1]
            * pred_meta_topk[i]["meter_per_pixel"].detach().cpu().numpy()
            * args["arl_img_size"][1]
        )
        dyaw = pred_pose[i][0][0][2]
        pred_locs_2.append([dlat, dlon, dyaw])

    pred_locs = []
    for i in range(len(pred_meta_topk)):
        yaw = pred_locs_1[i][2]
        c, s = np.cos(np.deg2rad(-yaw)), np.sin(np.deg2rad(-yaw))
        R = np.array([[c, -s], [s, c]])
        diff_shift = R @ np.array(pred_locs_2[i][:2])

        new_latitude = pred_locs_1[i][0] + (diff_shift[0] / r_earth) * (180 / np.pi)
        new_longitude = pred_locs_1[i][1] + (diff_shift[1] / r_earth) * (
            180 / np.pi
        ) / np.cos(pred_locs_1[i][0] * np.pi / 180)

        pred_locs.append([new_latitude, new_longitude])

    return pred_locs


def save_output(args, pred_metas, pred_locs, output_dir):

    folder_name = "{date:%Y-%m-%d-%H-%M-%S}".format(date=datetime.datetime.now())
    os.makedirs(os.path.join(output_dir, folder_name), exist_ok=True)

    save_path_qry = os.path.join(output_dir, folder_name, "query.png")
    qry_img = Image.open(args["qry_path"], "r")
    qry_img.save(save_path_qry)

    ref_imgs = []
    for i, pred_meta in enumerate(pred_metas):
        save_path_ref = os.path.join(
            output_dir, folder_name, "top_" + str(i + 1) + ".png"
        )
        ref_img = Image.open(
            os.path.join(args["data_root"], pred_metas[i]["file_name"]), "r"
        )
        ref_img.save(save_path_ref)
        ref_imgs.append(ref_img)

    factor = (ref_imgs[0].size[0] * len(pred_metas)) / float(qry_img.size[0])
    qry_img = qry_img.resize(
        (int(qry_img.size[0] * factor), int(qry_img.size[1] * factor))
    )
    board = np.ones(
        (
            ref_imgs[0].size[1] + qry_img.size[1],
            ref_imgs[0].size[0] * len(pred_metas)
            + ref_imgs[0].size[1]
            + qry_img.size[1],
            3,
        ),
        dtype=np.uint8,
    )
    for i in range(len(ref_imgs)):
        board[
            qry_img.size[1] :, ref_imgs[i].size[1] * i : ref_imgs[i].size[0] * (i + 1)
        ] = np.array(ref_imgs[i].convert("RGB"))

    board[: qry_img.size[1], : qry_img.size[0]] = np.array(qry_img)

    Lats, Longs, IDs = [], [], []

    for i, pred_loc in enumerate(pred_locs):
        lat, lon = pred_loc[0], pred_loc[1]
        Lats.append(lat)
        Longs.append(lon)
        IDs.append(i)

    df = pd.DataFrame({"ID": IDs, "Lat": Lats, "Long": Longs})

    fig = px.scatter_mapbox(
        df,
        lat="Lat",
        lon="Long",
        hover_name="ID",
        hover_data=["ID"],
        # color="Listed",
        # color_continuous_scale=color_scale,
        zoom=16,
        height=800,
        width=800,
    )
    fig.update_traces(marker=dict(size=15, color="red"))

    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    # fig.show()

    save_path_map = os.path.join(output_dir, folder_name, "open_street_map.png")
    fig.write_image(save_path_map)
    osm_img = (
        Image.open(save_path_map, "r")
        .resize((qry_img.size[1] + ref_img.size[0], qry_img.size[1] + ref_img.size[0]))
        .convert("RGB")
    )

    board[
        : osm_img.size[0],
        ref_img.size[0] * len(pred_metas) : ref_img.size[0] * len(pred_metas)
        + osm_img.size[0],
    ] = osm_img
    board_img = Image.fromarray(board, "RGB")
    draw = ImageDraw.Draw(board_img)
    font = ImageFont.truetype("./demo/Helvetica Bold.ttf", 200)
    draw.text((10, 10), f"Query Image", fill="black", font=font)
    draw.text((qry_img.size[0] + 10, 10), f"Prediction on OpenStreetMap", fill="black", font=font)
    font = ImageFont.truetype("./demo/Helvetica Bold.ttf", 150)
    for i in range(len(ref_imgs)):
        draw.text(
            (ref_imgs[i].size[1] * i + 10, qry_img.size[1] + 10),
            f"TOP #{i+1}",
            fill="white",
            font=font,
        )

    board_img.resize((board_img.size[0] // 5, board_img.size[1] // 5)).save(
        os.path.join(output_dir, folder_name, "results_summary.png")
    )

    save_path_latlon = os.path.join(output_dir, folder_name, "pose.txt")
    f = open(save_path_latlon, "w")
    f.write("Lat, Long\n")

    for i, pred_loc in enumerate(pred_locs):
        lat, lon = pred_loc[0], pred_loc[1]
        line = "{:.6f}, {:.6f}\n".format(
            lat,
            lon,
        )
        f.write(line)
    f.close

    print("[i] check ", os.path.join(output_dir, folder_name))
    print("[i] check ", os.path.join(output_dir, folder_name, "results_summary.png"))
    return


class PostProcess(nn.Module):

    @torch.no_grad()
    def forward(self, outputs):
        out_logits, out_bbox = (
            outputs["pred_logits"],
            outputs["pred_boxes"],
        )  # bs x num_quries x 4

        prob = torch.sigmoid(out_logits)
        scores = prob[..., :-1]

        x_c, y_c, c, s = out_bbox.unbind(-1)  # bs x num_quries
        yaw = torch.atan2(s, c)

        xs, ys = [], []
        for b in range(len(out_logits)):
            x = x_c[b]
            y = y_c[b]
            xs.append(x)
            ys.append(y)
        xs = torch.stack(xs, 0)
        ys = torch.stack(ys, 0)

        boxes = torch.stack([xs, ys, yaw], dim=-1)

        results = [{"scores": s, "boxes": b} for s, b in zip(scores, boxes)]
        return results
