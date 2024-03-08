import os
import torch
from torch import nn

from common.utils_loader import input_transform
from PIL import Image

import numpy as np

import os
import plotly.express as px
import pandas as pd


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
        arl_img = Image.open(os.path.join(args["db_root"], file_name), "r")
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


def save_output(args, pred_locs, output_dir):

    qry_name = (
        args["qry_path"].split("/")[-4] + "_" + args["qry_path"].split("/")[-1][:-4]
    )
    os.makedirs(os.path.join(output_dir, qry_name), exist_ok=True)

    save_path_qry = os.path.join(output_dir, qry_name, "qry.png")
    qry_img = Image.open(args["qry_path"], "r")
    qry_img.save(save_path_qry)

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
        # size="Size",
        zoom=16,
        height=800,
        width=800,
    )

    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    # fig.show()

    save_path_map = os.path.join(output_dir, qry_name, "res_map.png")
    fig.write_image(save_path_map)

    save_path_latlon = os.path.join(output_dir, qry_name, "pose.txt")
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

    print("[i] check ", os.path.join(output_dir, qry_name))
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
