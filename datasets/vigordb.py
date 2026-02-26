import os

import numpy as np
import torch
from PIL import Image

from common.utils_loader import input_transform


# Same loader from VIGOR, modified for pytorch
class VIGORDB(torch.utils.data.Dataset):
    def __init__(self, mode, args):
        super(VIGORDB, self).__init__()

        self.root = args["data_root"]

        self.transform_reference = input_transform(size=args["arl_img_size"])

        self.same_area = args["same_area"]

        # if self.same_area:
        #     self.city_list = ["NewYork", "Seattle", "SanFrancisco", "Chicago"]
        # else:
        #     self.city_list = ["SanFrancisco", "Chicago"]
        if self.same_area:
            self.city_list = ["NewYork", "SanFrancisco"]
        else:
            self.city_list = ["SanFrancisco"]

        self.arl_img_size = args["arl_img_size"]
        self.raw_arl_img_size = (640, 640)

        self.arl_zoom_ratio = self.raw_arl_img_size[0] / self.arl_img_size[0]
        self.meter_per_pixel_dict = {
            "Chicago": 0.111,
            "NewYork": 0.113,
            "SanFrancisco": 0.118,
            "Seattle": 0.101,
        }  # based on SliceMatch

        self.label_root = "splits__corrected"
        self.make_slice_match_sample_list()

    def make_slice_match_sample_list(self):
        self.sat_list = []
        self.sat_index_dict = {}
        idx = 0
        for city in self.city_list:
            sat_list_fname = os.path.join(
                self.root, self.label_root, city, "satellite_list.txt"
            )
            with open(sat_list_fname, "r") as file:
                for line in file.readlines():
                    self.sat_list.append(
                        os.path.join(
                            self.root, city, "satellite", line.replace("\n", "")
                        )
                    )
                    self.sat_index_dict[line.replace("\n", "")] = idx
                    idx += 1
        self.sat_list = np.array(self.sat_list)
        self.sat_data_size = len(self.sat_list)

    def __getitem__(self, index):
        file_name = self.sat_list[index]

        arl_img = Image.open(file_name).convert("RGB")
        img_ref = self.transform_reference(arl_img)

        city = file_name.split("/")[-3]
        fn = file_name.split("/")[-1]
        lat, lon = fn.split("_")[1], fn.split("_")[2][:-4]
        meta_info = {
            "lat": float(lat),
            "lon": float(lon),
            "yaw": 0,
            "meter_per_pixel": self.meter_per_pixel_dict[city],
            "file_name": file_name,
        }

        return (img_ref, torch.tensor(index), meta_info)

    def __len__(self):
        return len(self.sat_list)
