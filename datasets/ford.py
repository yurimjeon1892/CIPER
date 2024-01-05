import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from PIL import Image
import numpy as np
import os, yaml

from common.utils_loader import gps2utm
from .ford_cfgnode import CfgNode


# For the Ford dataset coordinates:
# x--> North, y --> east, z --> down
# North direction as 0-degree, clockwise as positive.


class Ford(torch.utils.data.Dataset):
    def __init__(self, mode, args):
        super(Ford, self).__init__()

        self.mode = mode
        self.root = args["data_root"]
        self.log_idx = args["log_idx"]

        self.arl_img_size = args["arl_img_size"]
        self.grd_img_size = args["grd_img_size"]

        shift_range_lat = float(args["shift_range_lat"])
        shift_range_lon = float(args["shift_range_lon"])
        rotation_range = float(args["rotation_range"])

        self.shift_range_meters_lat = shift_range_lat  # in terms of meters
        self.shift_range_meters_lon = shift_range_lon  # in terms of meters
        self.meters_per_pixel = 0.22
        self.shift_range_pixels_lat = shift_range_lat / float(
            self.meters_per_pixel
        )  # in terms of pixels
        self.shift_range_pixels_lon = shift_range_lon / float(
            self.meters_per_pixel
        )  # in terms of pixels

        self.rotation_range = rotation_range  # in terms of degree

        self.make_sample_list()

    def make_sample_list(self):
        satmap_dir = "SatelliteMaps_18"
        pose_file_dir = "Calibration-V2/V2/"
        FL_ex = "cameraFrontLeft_body.yaml"
        FL_in = "cameraFrontLeftIntrinsics.yaml"

        if self.mode == "train":
            logs = train_logs
            logs_img_inds = train_logs_img_inds
            data_file = "grd_sat_quaternion_latlon.txt"
        else:
            logs = test_logs
            logs_img_inds = test_logs_img_inds
            data_file = "grd_sat_quaternion_latlon_test.txt"

        file_name = []
        idx = self.log_idx
        # for idx in range(len(logs)):
        log = logs[idx]
        img_inds = logs_img_inds[idx]
        FL_dir = os.path.join(self.root, log, log.replace("/", "-") + "-FL")

        if self.mode == "train":
            with open(os.path.join(self.root, log, data_file), "r") as f:
                lines = f.readlines()
                lines = [lines[ind] for ind in img_inds]
                # lines = f.readlines()[img_inds]
                for line in lines:
                    (
                        grd_name,
                        q0,
                        q1,
                        q2,
                        q3,
                        g_lat,
                        g_lon,
                        s_lat,
                        s_lon,
                    ) = line.strip().split(" ")
                    grd_file_FL = os.path.join(
                        self.root, log, FL_dir, grd_name.replace(".txt", ".png")
                    )
                    sat_file = os.path.join(
                        self.root, log, satmap_dir, s_lat + "_" + s_lon + ".png"
                    )
                    file_name.append(
                        [
                            grd_file_FL,
                            float(q0),
                            float(q1),
                            float(q2),
                            float(q3),
                            float(g_lat),
                            float(g_lon),
                            float(s_lat),
                            float(s_lon),
                            sat_file,
                        ]
                    )
        elif "valid" in self.mode:
            with open(os.path.join(self.root, log, data_file), "r") as f:
                lines = f.readlines()
                # if whole == 0:
                #     lines = [lines[ind] for ind in img_inds]
                # lines = f.readlines()[img_inds]
                for line in lines:
                    (
                        grd_name,
                        q0,
                        q1,
                        q2,
                        q3,
                        g_lat,
                        g_lon,
                        s_lat,
                        s_lon,
                        gt_shift_u,
                        gt_shift_v,
                        theta,
                    ) = line.strip().split(" ")
                    grd_file_FL = os.path.join(
                        self.root, log, FL_dir, grd_name.replace(".txt", ".png")
                    )
                    sat_file = os.path.join(
                        self.root, log, satmap_dir, s_lat + "_" + s_lon + ".png"
                    )
                    file_name.append(
                        [
                            grd_file_FL,
                            float(q0),
                            float(q1),
                            float(q2),
                            float(q3),
                            float(g_lat),
                            float(g_lon),
                            float(s_lat),
                            float(s_lon),
                            sat_file,
                            float(gt_shift_u),
                            float(gt_shift_v),
                            float(theta),
                        ]
                    )

        self.file_name = file_name

        self.lat0 = 42.29424422604817  # 08-04-Log0-img0

        with open(os.path.join(self.root, pose_file_dir, FL_ex), "r") as f:
            cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
            cfg_FL_ex = CfgNode(cfg_dict)

        qx = cfg_FL_ex.transform.rotation.x
        qy = cfg_FL_ex.transform.rotation.y
        qz = cfg_FL_ex.transform.rotation.z
        qw = cfg_FL_ex.transform.rotation.w

        FLx, FLy, FLz = (
            cfg_FL_ex.transform.translation.x,
            cfg_FL_ex.transform.translation.y,
            cfg_FL_ex.transform.translation.z,
        )
        self.T_FL = np.array([FLx, FLy, FLz]).reshape(3).astype(np.float32)
        self.R_FL = qvec2rotmat([qw, qx, qy, qz]).astype(np.float32)
        # from camera coordinates to body coordinates
        # Xb = R_FL @ Xc + T_FL

        with open(os.path.join(self.root, pose_file_dir, FL_in), "r") as f:
            cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
            cfg_FL_in = CfgNode(cfg_dict)

        self.K_FL = np.array(cfg_FL_in.K, dtype=np.float32).reshape([3, 3])
        self.H_FL = 860
        self.W_FL = 1656

        # self.H = 256
        # self.W = 1024

        self.K_FL[0] = self.K_FL[0] / self.W_FL * self.grd_img_size[1]
        self.K_FL[1] = self.K_FL[1] / self.H_FL * self.grd_img_size[0]

        self.sidelength = self.arl_img_size[0]
        self.satmap_sidelength_meters = self.sidelength * self.meters_per_pixel
        self.satmap_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.grdimage_transform = transforms.Compose(
            [
                transforms.Resize(size=self.grd_img_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.file_name = self.file_name[:100]

    def __getitem__(self, index):
        if self.mode == "train" or self.mode == "valid_same":
            if self.mode == "train":
                (
                    grd_name,
                    q0,
                    q1,
                    q2,
                    q3,
                    g_lat,
                    g_lon,
                    s_lat,
                    s_lon,
                    sat_name,
                ) = self.file_name[index]
            else:
                (
                    grd_name,
                    q0,
                    q1,
                    q2,
                    q3,
                    g_lat,
                    g_lon,
                    s_lat,
                    s_lon,
                    sat_name,
                    gt_shift_u,
                    gt_shift_v,
                    theta,
                ) = self.file_name[index]

            # =================== read ground image ===================================
            grd_img = Image.open(grd_name).convert("RGB")

            # =================== read satellite map ==================================
            sat_map = Image.open(sat_name).convert("RGB")

            # =================== initialize some required variables ==================
            # Xc = np.array([0, 0, 0]).reshape(3)
            # Rw = qvec2rotmat([float(q0), float(q1), float(q2), float(q3)])
            # # body frame to world frame: Xw = Rw @ Xb + Tw  (Tw are all zeros)
            # Xw = Rw @ (self.R_FL @ Xc + self.T_FL)  # North (up) --> X, East (right) --> Y
            # # camera location represented in world coordinates,
            # # world coordinates is centered at the body coordinates, but with X pointing north, Y pointing east, Z pointing down

            g_x, g_y = gps2utm(float(g_lat), float(g_lon), float(s_lat))
            s_x, s_y = gps2utm(float(s_lat), float(s_lon), float(s_lat))
            # x, y here are the x, y under gps/utm coordinates, x pointing right and y pointing up

            b_delta_u = (
                g_x - s_x
            ) / self.meters_per_pixel  # relative u shift of body frame with respect to satellite image center
            b_delta_v = (
                -(g_y - s_y) / self.meters_per_pixel
            )  # relative v shift of body frame with respect to satellite image center

            sat_align_body_loc = sat_map.transform(
                sat_map.size,
                Image.AFFINE,
                (1, 0, b_delta_u, 0, 1, b_delta_v),
                resample=Image.BILINEAR,
            )
            # Homography is defined on from target pixel to source pixel
            roll, pitch, yaw = qvec2angle(q0, q1, q2, q3)  # in terms of degree
            sat_align_body_loc_orien = sat_align_body_loc.rotate(yaw)

            # =================== add random translation & rotation ===================
            grd_img = self.grdimage_transform(grd_img)

            if self.mode == "train":
                theta = np.random.uniform(-1, 1)
            sat_rand_rot = sat_align_body_loc_orien.rotate(theta * self.rotation_range)

            if self.mode == "train":
                # random shift
                gt_shift_u = np.random.uniform(
                    -1, 1
                )  # --> right (east) as positive, vertical to the heading, lateral
                gt_shift_v = np.random.uniform(
                    -1, 1
                )  # --> down (south) as positive, parallel to the heading, longitudinal

            sat_rand_rot_rand_shift = sat_rand_rot.transform(
                sat_align_body_loc_orien.size,
                Image.AFFINE,
                (
                    1,
                    0,
                    gt_shift_u * self.shift_range_pixels_lat,
                    0,
                    1,
                    gt_shift_v * self.shift_range_pixels_lon,
                ),
                resample=Image.BILINEAR,
            )

            sat_img = TF.center_crop(sat_rand_rot_rand_shift, self.sidelength)
            sat_img = self.satmap_transform(sat_img)

            # =================== make target dict ====================================
            tgt_x = -(gt_shift_v * self.shift_range_pixels_lon) / self.arl_img_size[1]
            tgt_y = -(gt_shift_u * self.shift_range_pixels_lat) / self.arl_img_size[0]

            tgt_rad = np.deg2rad(theta * self.rotation_range + 180.0)
            tgt_cos = np.cos(tgt_rad)
            tgt_sin = np.sin(tgt_rad)

            target = {
                "boxes": torch.tensor(np.array([[tgt_x, tgt_y, tgt_cos, tgt_sin]])),
                "labels": torch.tensor([0]),
                "orig_size": torch.as_tensor(
                    np.array([int(self.arl_img_size[0]), int(self.arl_img_size[1])])
                ),
                "meter_per_pixel": torch.as_tensor(np.array([self.meters_per_pixel])),
                "R_FL": torch.as_tensor(np.array([self.R_FL])),
                "T_FL": torch.as_tensor(np.array([self.T_FL])),
                # "grd_name": torch.tensor([grd_name])
            }

            return grd_img, sat_img, target

        elif self.mode == "valid_same_ref":
            (
                grd_name,
                q0,
                q1,
                q2,
                q3,
                g_lat,
                g_lon,
                s_lat,
                s_lon,
                sat_name,
                gt_shift_u,
                gt_shift_v,
                theta,
            ) = self.file_name[index]

            g_x, g_y = gps2utm(float(g_lat), float(g_lon), float(s_lat))
            s_x, s_y = gps2utm(float(s_lat), float(s_lon), float(s_lat))
            # x, y here are the x, y under gps/utm coordinates, x pointing right and y pointing up

            b_delta_u = (
                g_x - s_x
            ) / self.meters_per_pixel  # relative u shift of body frame with respect to satellite image center
            b_delta_v = (
                -(g_y - s_y) / self.meters_per_pixel
            )  # relative v shift of body frame with respect to satellite image center

            # =================== read satellite map ==================================
            sat_map = Image.open(sat_name).convert("RGB")

            # =================== initialize some required variables ==================
            sat_align_body_loc = sat_map.transform(
                sat_map.size,
                Image.AFFINE,
                (1, 0, b_delta_u, 0, 1, b_delta_v),
                resample=Image.BILINEAR,
            )
            # Homography is defined on from target pixel to source pixel
            roll, pitch, yaw = qvec2angle(q0, q1, q2, q3)  # in terms of degree
            sat_align_body_loc_orien = sat_align_body_loc.rotate(yaw)

            # random shift
            # gt_shift_u = np.random.uniform(-1, 1)  # --> right (east) as positive, vertical to the heading, lateral
            # gt_shift_v = np.random.uniform(-1, 1)  # --> down (south) as positive, parallel to the heading, longitudinal

            # =================== add random translation & rotation ===================
            sat_rand_rot = sat_align_body_loc_orien.rotate(theta * self.rotation_range)

            sat_rand_rot_rand_shift = sat_rand_rot.transform(
                sat_align_body_loc_orien.size,
                Image.AFFINE,
                (
                    1,
                    0,
                    gt_shift_u * self.shift_range_pixels_lat,
                    0,
                    1,
                    gt_shift_v * self.shift_range_pixels_lon,
                ),
                resample=Image.BILINEAR,
            )

            # theta = np.random.uniform(-1, 1)

            sat_img = TF.center_crop(sat_rand_rot_rand_shift, self.sidelength)
            sat_img = self.satmap_transform(sat_img)

            return sat_img, torch.tensor(index), 0

        elif self.mode == "valid_same_qry":
            (
                grd_name,
                q0,
                q1,
                q2,
                q3,
                g_lat,
                g_lon,
                s_lat,
                s_lon,
                sat_name,
                gt_shift_u,
                gt_shift_v,
                theta,
            ) = self.file_name[index]

            grd_img = Image.open(grd_name).convert("RGB")
            grd_img = self.grdimage_transform(grd_img)

            return grd_img, torch.tensor(index), torch.tensor(index)

        else:
            print("not implemented!!")
            raise Exception

    def __len__(self):
        return len(self.file_name)


train_logs = [
    "2017-10-26/V2/Log1",
    "2017-10-26/V2/Log2",
    "2017-08-04/V2/Log3",
    "2017-10-26/V2/Log4",
    "2017-08-04/V2/Log5",
    "2017-08-04/V2/Log6",
]

train_logs_img_inds = [
    list(range(4500, 8500)),
    list(range(3150)) + list(range(6000, 9200)) + list(range(11000, 15000)),
    list(range(1500)),
    list(range(7466)),
    list(range(3200)) + list(range(5300, 9900)) + list(range(10500, 11130)),
    list(range(1000, 3500)) + list(range(4500, 5000)) + list(range(7000, 7857)),
]

test_logs = [
    "2017-08-04/V2/Log1",
    "2017-08-04/V2/Log2",
    "2017-08-04/V2/Log3",
    "2017-08-04/V2/Log4",
    "2017-10-26/V2/Log5",
    "2017-10-26/V2/Log6",
]
test_logs_img_inds = [
    list(range(100, 200)) + list(range(5000, 5500)) + list(range(7000, 8500)),
    list(range(2500, 3000)) + list(range(8500, 10500)) + list(range(12500, 13727)),
    list(range(3500, 5000)),
    list(range(1500, 2500)) + list(range(4000, 4500)) + list(range(7000, 9011)),
    list(range(3500)),
    list(range(2000, 2500)) + list(range(3500, 4000)),
]


def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def qvec2angle(q0, q1, q2, q3):
    roll = (
        np.arctan2(2.0 * (q3 * q2 + q0 * q1), 1.0 - 2.0 * (q1 * q1 + q2 * q2))
        / np.pi
        * 180
    )
    pitch = np.arcsin(2.0 * (q2 * q0 - q3 * q1)) / np.pi * 180
    yaw = (
        np.arctan2(2.0 * (q3 * q0 + q1 * q2), -1.0 + 2.0 * (q0 * q0 + q1 * q1))
        / np.pi
        * 180
    )
    return roll, pitch, yaw
