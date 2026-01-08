import os

import natsort
import numpy as np
import torch
from PIL import Image

from common.utils_loader import input_transform

from .kitti_func import get_meter_per_pixel


class KITTIDB(torch.utils.data.Dataset):
    def __init__(self, mode, args):
        super(KITTIDB, self).__init__()

        self.root = args["data_root"]

        self.transform_reference = input_transform(size=args["arl_img_size"])
        self.arl_img_size = args["arl_img_size"]

        self.meter_per_pixel = get_meter_per_pixel(scale=1)
        self.make_sample_list()

    def make_sample_list(self):
        self.sample_list = []

        date_dirs = natsort.natsorted(os.listdir(os.path.join(self.root, "satellite")))
        for date_dir in date_dirs:
            if not os.path.isdir(os.path.join(self.root, "satellite", date_dir)):
                continue
            drive_dirs = natsort.natsorted(
                os.listdir(os.path.join(self.root, "satellite", date_dir))
            )
            for drive_dir in drive_dirs:
                if not os.path.isdir(
                    os.path.join(self.root, "satellite", date_dir, drive_dir)
                ):
                    continue
                if not os.path.isdir(
                    os.path.join(self.root, "raw", date_dir, drive_dir)
                ):
                    continue
                file_names = natsort.natsorted(
                    os.listdir(
                        os.path.join(self.root, "satellite", date_dir, drive_dir)
                    )
                )
                for file_name in file_names:
                    if file_name[-4:] != ".png":
                        continue
                    if int(file_name[:-4]) % 10 != 0:  # sampling!
                        continue
                    self.sample_list.append(
                        os.path.join("satellite", date_dir, drive_dir, file_name)
                    )

        print("[i] database loaded, size: {}".format(len(self.sample_list)))

    def __getitem__(self, index):
        file_name = self.sample_list[index]
        arl_img_name = os.path.join(self.root, file_name)
        try:
            arl_img = Image.open(arl_img_name, "r")
            arl_img = arl_img.convert("RGB")
        except:
            FileNotFoundError(f"{arl_img_name} not exists")
        arl_img = self.transform_reference(arl_img)

        date_dir, drive_dir, file_name_ = (
            file_name.split("/")[1],
            file_name.split("/")[2],
            file_name.split("/")[3],
        )

        oxts_file_name = os.path.join(
            self.root,
            "raw",
            date_dir,
            drive_dir,
            "oxts/data",
            file_name_.lower().replace(".png", ".txt"),
        )
        with open(oxts_file_name, "r") as f:
            content = f.readline().split(" ")
        lat, lon = float(content[0]), float(content[1])
        yaw = float(content[5]) / np.pi * 180 - 90.0

        meta_info = {
            "lat": lat,
            "lon": lon,
            "yaw": yaw,
            "meter_per_pixel": self.meter_per_pixel,
            "file_name": file_name,
        }

        return (
            arl_img,
            torch.tensor(index),
            meta_info,
        )

    def __len__(self):
        return len(self.sample_list)
