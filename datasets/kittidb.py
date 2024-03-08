import torch

from PIL import Image
import numpy as np
import os
import natsort

from common.utils_loader import input_transform
from .kitti_func import get_meter_per_pixel


class KITTIDB(torch.utils.data.Dataset):
    def __init__(self, mode, args):
        super(KITTIDB, self).__init__()

        self.root = args["db_root"]

        self.transform_reference = input_transform(size=args["arl_img_size"])
        self.arl_img_size = args["arl_img_size"]

        self.meter_per_pixel = get_meter_per_pixel(scale=1)
        self.make_sample_list()

    def make_sample_list(self):

        self.sample_list = []

        drive_dirs = natsort.natsorted(os.listdir(self.root))
        for drive_dir in drive_dirs:
            file_names = natsort.natsorted(
                os.listdir(os.path.join(self.root, drive_dir))
            )
            for file_name in file_names:
                if file_name[-4:] != ".png":
                    continue
                self.sample_list.append(os.path.join(drive_dir, file_name))
            break

        print("[i] data base loaded, size: {}".format(len(self.sample_list)))

    def __getitem__(self, index):

        file_name = self.sample_list[index]
        arl_img_name = os.path.join(self.root, file_name)
        try:
            arl_img = Image.open(arl_img_name, "r")
            arl_img = arl_img.convert("RGB")
        except:
            FileNotFoundError(f"{arl_img_name} not exists")
        arl_img = self.transform_reference(arl_img)

        date, roots = self.root.split("/")[-1], self.root.split("/")[:-2]

        drive_dir, file_name_ = file_name.split("/")[0], file_name.split("/")[1]

        oxts_file_name = os.path.join(
            "/",
            *roots,
            "raw",
            date,
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
