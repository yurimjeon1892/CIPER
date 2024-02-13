import torch
from PIL import Image
import numpy as np
import os
import random
from sklearn.utils import shuffle

from common.utils_loader import input_transform, input_transform_fov


# Same loader from VIGOR, modified for pytorch
class VIGOR(torch.utils.data.Dataset):
    def __init__(self, mode, args):
        super(VIGOR, self).__init__()

        self.mode = mode
        self.root = args["data_root"]

        if args["fov"] != 0:
            self.transform_query = input_transform_fov(
                size=args["grd_img_size"], fov=args["fov"]
            )
        else:
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

        self.city_list = ["Seattle"]

        self.arl_img_size = args["arl_img_size"]
        self.rotation_range = args["rotation_range"]
        self.raw_arl_img_size = (640, 640)

        self.pos_only = args["pos_only"]

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

            if self.same_area:
                if self.mode == "train":
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
            if not self.same_area:
                label_fname = os.path.join(
                    "datasets/splits/vigor",
                    self.label_root,
                    city,
                    "pano_label_balanced__corrected.txt",
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

        for rand_state in range(20):
            self.grd_list, self.label, self.delta = shuffle(
                self.grd_list, self.label, self.delta, random_state=rand_state
            )

        self.data_size = 50

        # self.data_size = int(len(self.grd_list))
        self.grd_list = self.grd_list[: self.data_size]
        self.label = self.label[: self.data_size]
        self.delta = self.delta[: self.data_size]
        self.label = np.array(self.label)
        self.delta = np.array(self.delta)

        self.sat_cover_list = list(self.sat_cover_dict.keys())

    def get_grd_sat_img_pair(self, idx):

        # full ground panorama
        try:
            grd = Image.open(os.path.join(self.root, self.grd_list[idx]))
            grd = grd.convert("RGB")
        except:
            print("unreadable image")
            print(os.path.join(self.root, self.grd_list[idx]))
            grd = Image.new(
                "RGB", (320, 640)
            )  # if the image is unreadable, use a blank image
        # grd = self.grdimage_transform(grd)
        grd = self.transform_query(grd)

        # generate a random rotation
        rotation = np.random.uniform(low=-1.0, high=1.0)  #
        rotation_angle = rotation * self.rotation_range
        grd = torch.roll(
            grd,
            (
                torch.round(
                    torch.as_tensor(rotation_angle / 180) * grd.size()[2] / 2
                ).int()
            ).item(),
            dims=2,
        )

        # satellite
        if self.pos_only:  # load positives only
            pos_index = 0
            sat = Image.open(os.path.join(self.sat_list[self.label[idx][pos_index]]))
            [row_offset, col_offset] = self.delta[
                idx, pos_index
            ]  # delta = [delta_lat, delta_lon]
        else:  # load positives and semi-positives
            col_offset = 320
            row_offset = 320
            while (
                np.abs(col_offset) >= 320 or np.abs(row_offset) >= 320
            ):  # do not use the semi-positives where GT location is outside the image
                pos_index = random.randint(0, 3)
                sat = Image.open(
                    os.path.join(self.sat_list[self.label[idx][pos_index]])
                )
                [row_offset, col_offset] = self.delta[
                    idx, pos_index
                ]  # delta = [delta_lat, delta_lon]

        sat = sat.convert("RGB")
        sat = self.transform_reference(sat)

        # # groundtruth location on the aerial image
        # gt_shift_y = row_offset / height * 4  # -L/4 ~ L/4  -1 ~ 1
        # gt_shift_x = -col_offset / width * 4  #

        if "NewYork" in self.grd_list[idx]:
            city = "NewYork"
        elif "Seattle" in self.grd_list[idx]:
            city = "Seattle"
        elif "SanFrancisco" in self.grd_list[idx]:
            city = "SanFrancisco"
        elif "Chicago" in self.grd_list[idx]:
            city = "Chicago"

        gt_shift_y = row_offset
        gt_shift_x = -col_offset

        tgt_y = (gt_shift_x / self.arl_zoom_ratio) / self.arl_img_size[1]
        tgt_x = (gt_shift_y / self.arl_zoom_ratio) / self.arl_img_size[0]

        tgt_rad = np.deg2rad(-rotation_angle)
        tgt_cos = np.cos(tgt_rad)
        tgt_sin = np.sin(tgt_rad)

        target = {
            "boxes": torch.tensor([[tgt_x, tgt_y, tgt_cos, tgt_sin]]),
            "labels": torch.tensor([0]),
            "orig_size": torch.as_tensor(
                [int(self.arl_img_size[0]), int(self.arl_img_size[1])]
            ),
            "arl_zoom_ratio": torch.tensor([self.arl_zoom_ratio]),
            "meter_per_pixel": torch.tensor([self.meter_per_pixel_dict[city]]),
        }

        return grd, sat, target

    def __getitem__(self, index):
        if self.mode == "train" or self.mode == "valid_same":

            return self.get_grd_sat_img_pair(index)

        elif self.mode == "valid_same_ref":

            # satellite
            if self.pos_only:  # load positives only
                pos_index = 0
                sat = Image.open(
                    os.path.join(self.sat_list[self.label[index][pos_index]])
                )
                [row_offset, col_offset] = self.delta[
                    index, pos_index
                ]  # delta = [delta_lat, delta_lon]
            else:  # load positives and semi-positives
                col_offset = 320
                row_offset = 320
                while (
                    np.abs(col_offset) >= 320 or np.abs(row_offset) >= 320
                ):  # do not use the semi-positives where GT location is outside the image
                    pos_index = random.randint(0, 3)
                    sat = Image.open(
                        os.path.join(self.sat_list[self.label[index][pos_index]])
                    )
                    [row_offset, col_offset] = self.delta[
                        index, pos_index
                    ]  # delta = [delta_lat, delta_lon]

            sat = sat.convert("RGB")
            sat = self.transform_reference(sat)

            return sat, torch.tensor(index), 0

        elif self.mode == "valid_same_qry":

            # full ground panorama
            try:
                grd = Image.open(os.path.join(self.root, self.grd_list[index]))
                grd = grd.convert("RGB")
            except:
                print("unreadable image")
                print(os.path.join(self.root, self.grd_list[index]))
                grd = Image.new(
                    "RGB", (320, 640)
                )  # if the image is unreadable, use a blank image
            # grd = self.grdimage_transform(grd)
            grd = self.transform_query(grd)

            # generate a random rotation
            rotation = np.random.uniform(low=-1.0, high=1.0)  #
            rotation_angle = rotation * self.rotation_range
            grd = torch.roll(
                grd,
                (
                    torch.round(
                        torch.as_tensor(rotation_angle / 180) * grd.size()[2] / 2
                    ).int()
                ).item(),
                dims=2,
            )

            return grd, torch.tensor(index), torch.tensor(self.label[index][0])
        else:
            print("not implemented!!")
            raise Exception

    # def __len__(self):
    #     if "train" in self.mode:
    #         return (
    #             len(self.sat_cover_list) * 2
    #         )  # one aerial image has 2 positive queries
    #     elif "valid_same_ref" in self.mode:
    #         return len(self.sat_list)
    #     elif "valid_same_qry" in self.mode:
    #         return len(self.grd_list)
    #     elif "valid_same" in self.mode:
    #         return (
    #             len(self.sat_cover_list) * 2
    #         )  # one aerial image has 2 positive queries
    #     else:
    #         print("not implemented!")
    #         raise Exception
    def __len__(self):
        return 50
