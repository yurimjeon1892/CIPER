import torch
from PIL import Image
import numpy as np
import os
import random

from common.utils_loader import input_transform


# Same loader from VIGOR, modified for pytorch
class VIGOR(torch.utils.data.Dataset):
    def __init__(self, mode, args):
        super(VIGOR, self).__init__()

        self.mode = mode
        self.root = args["data_root"]

        self.transform_query = input_transform(size=args["grd_img_size"])
        self.transform_reference = input_transform(size=args["arl_img_size"])

        self.same_area = args["same_area"]

        if self.same_area:
            if self.mode == "train":
                self.city_list = ["NewYork", "Seattle", "SanFrancisco", "Chicago"]
            else:
                self.city_list = ["NewYork", "Seattle", "SanFrancisco", "Chicago"]
        else:
            if self.mode == "train":
                self.city_list = ["NewYork", "Seattle"]
            else:
                self.city_list = ["SanFrancisco", "Chicago"]

        self.arl_img_size = args["arl_img_size"]
        self.rotation_range = args["rotation_range"]
        self.fov_ratio = args["fov"] / 360
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
                "datasets/splits/vigor", self.label_root, city, "satellite_list.txt"
            )
            with open(sat_list_fname, "r") as file:
                for line in file.readlines():
                    self.sat_list.append(
                        os.path.join(
                            self.root,
                            city,
                            "satellite",
                            line.replace("\n", ""),
                        )
                    )
                    self.sat_index_dict[line.replace("\n", "")] = idx
                    idx += 1
        self.sat_list = np.array(self.sat_list)
        self.sat_data_size = len(self.sat_list)

        self.grd_list = []
        self.label = []
        self.sat_cover_dict = {}
        self.delta = []
        self.meter_per_pixel_list = []
        idx = 0
        for city in self.city_list:
            # load train panorama list
            if not self.same_area:
                label_fname = os.path.join(
                    "datasets/splits/vigor",
                    self.label_root,
                    city,
                    "pano_label_balanced__corrected.txt",
                )
            elif self.mode == "train":
                label_fname = os.path.join(
                    "datasets/splits/vigor",
                    self.label_root,
                    city,
                    "same_area_balanced_train__corrected.txt",
                )
            else:
                label_fname = os.path.join(
                    "datasets/splits/vigor",
                    self.label_root,
                    city,
                    "same_area_balanced_test__corrected.txt",
                )
            with open(label_fname, "r") as file:
                for line in file.readlines():
                    data = np.array(line.split(" "))
                    label = []
                    for i in [1, 4, 7, 10]:
                        label.append(self.sat_index_dict[data[i]])
                    label = np.array(label).astype(int)
                    delta = np.array(
                        [data[2:4], data[5:7], data[8:10], data[11:13]]
                    ).astype(float)
                    self.grd_list.append(
                        os.path.join(self.root, city, "panorama", data[0])
                    )
                    self.label.append(label)
                    self.delta.append(delta)
                    if not label[0] in self.sat_cover_dict:
                        self.sat_cover_dict[label[0]] = [idx]
                    else:
                        self.sat_cover_dict[label[0]].append(idx)
                    self.meter_per_pixel_list.append(self.meter_per_pixel_dict[city])
                    idx += 1

        self.data_size = len(self.grd_list)
        self.label = np.array(self.label)
        self.delta = np.array(self.delta)
        self.sat_cover_list = list(self.sat_cover_dict.keys())

    def prep_gt(self, gt_shift_x, gt_shift_y, theta, meter_per_pixel):
        tgt_y = (gt_shift_x / self.arl_zoom_ratio) / self.arl_img_size[1]
        tgt_x = (gt_shift_y / self.arl_zoom_ratio) / self.arl_img_size[0]

        tgt_rad = np.deg2rad(theta + 180.0)
        tgt_cos = np.cos(tgt_rad)
        tgt_sin = np.sin(tgt_rad)

        target = {
            "boxes": torch.tensor([[tgt_x, tgt_y, tgt_cos, tgt_sin]]),
            "labels": torch.tensor([0]),
            "orig_size": torch.as_tensor(
                [int(self.arl_img_size[0]), int(self.arl_img_size[1])]
            ),
            "arl_zoom_ratio": torch.tensor([self.arl_zoom_ratio]),
            "meter_per_pixel": torch.tensor([meter_per_pixel]),
        }
        return target

    def __getitem__(self, index):
        if self.mode == "train" or self.mode == "valid_same":
            idx = random.choice(
                self.sat_cover_dict[
                    self.sat_cover_list[index % len(self.sat_cover_list)]
                ]
            )

            grd_img = Image.open(os.path.join(self.root, self.grd_list[idx]))
            grd_img = np.array(grd_img)
            raw_width = grd_img.shape[1]
            new_width = raw_width * self.fov_ratio
            grd_img_new = grd_img[:, int((raw_width - new_width) / 2):int((raw_width + new_width) / 2), :]
            grd_img = self.transform_query(grd_img_new)

            # generate a random rotation
            rotation = np.random.uniform(low=-1.0, high=1.0)  #
            rotation_angle = rotation * self.rotation_range
            grd_img = torch.roll(
                grd_img,
                (
                    torch.round(
                        torch.as_tensor(rotation_angle / 180) * grd_img.size()[2] / 2
                    ).int()
                ).item(),
                dims=2,
            )

            arl_img = Image.open(self.sat_list[self.label[idx][0]]).convert("RGB")
            arl_img = self.transform_reference(arl_img)

            gt_shift_x = -self.delta[idx, 0][1]
            gt_shift_y = self.delta[idx, 0][0]

            meter_per_pixel = self.meter_per_pixel_list[idx]

            target = self.prep_gt(
                gt_shift_x, gt_shift_y, rotation_angle, meter_per_pixel
            )

            return grd_img, arl_img, target

        elif self.mode == "valid_same_ref":
            arl_img = Image.open(self.sat_list[index]).convert("RGB")
            arl_img = self.transform_reference(arl_img)

            return arl_img, torch.tensor(index), 0

        elif self.mode == "valid_same_qry":
            grd_img = Image.open(self.grd_list[index])
            grd_img = self.transform_query(grd_img)

            # generate a random rotation
            rotation = np.random.uniform(low=-1.0, high=1.0)  #
            rotation_angle = rotation * self.rotation_range
            grd_img = torch.roll(
                grd_img,
                (
                    torch.round(
                        torch.as_tensor(rotation_angle / 180) * grd_img.size()[2] / 2
                    ).int()
                ).item(),
                dims=2,
            )

            return grd_img, torch.tensor(index), torch.tensor(self.label[index][0])
        else:
            print("not implemented!!")
            raise Exception

    def __len__(self):
        if "train" in self.mode:
            return (
                len(self.sat_cover_list) * 2
            )  # one aerial image has 2 positive queries
        elif "valid_same_ref" in self.mode:
            return len(self.sat_list)
        elif "valid_same_qry" in self.mode:
            return len(self.grd_list)
        elif "valid_same" in self.mode:
            return (
                len(self.sat_cover_list) * 2
            )  # one aerial image has 2 positive queries
        else:
            print("not implemented!")
            raise Exception


def get_aerial_and_deltas(combination_dir):
    data_dict = {}
    with open(combination_dir, "r") as file:
        for line in file.readlines():
            data = line.split(" ")
            data_list = []
            for idx in range(4):
                data_list.append(
                    (
                        data[3 * idx + 1],
                        float(data[3 * idx + 2]),
                        float(data[3 * idx + 3]),
                    )
                )
            data_dict[data[0]] = data_list
    return data_dict
