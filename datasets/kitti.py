import torch
from torchvision import transforms
import torchvision.transforms.functional as TF

from PIL import Image
import numpy as np
import os

from common.utils_loader import input_transform
from .kitti_func import get_meter_per_pixel, CameraGPS_shift_left
    
class KITTI(torch.utils.data.Dataset):
    def __init__(self, mode, args):
        super(KITTI, self).__init__()
        
        self.mode = mode
        self.root = args["data_root"]
        
        self.transform_query = input_transform(size=args["grnd_img_size"])        
        self.transform_reference = input_transform(size=args["arl_img_size"])
        
        self.arl_img_size = args["arl_img_size"]
        self.grnd_img_size = args["grnd_img_size"] # 256, 1024 -> need to be reduced (320, 640 maybe? )
        
        self.arl_zoom_ratio = args["arl_zoom_ratio"]

        shift_range_lat = float(args["shift_range_lat"])
        shift_range_lon = float(args["shift_range_lon"])
        rotation_range = float(args["rotation_range"])
        
        self.down_ratio = float(args["down_ratio"])
        self.spare_pixel = int(args["spare_pixel"])

        self.meter_per_pixel = get_meter_per_pixel(scale=1/self.arl_zoom_ratio)
        self.shift_range_meters_lat = shift_range_lat  # in terms of meters
        self.shift_range_meters_lon = shift_range_lon  # in terms of meters
        self.shift_range_pixels_lat = shift_range_lat / float(self.meter_per_pixel)  # shift range is in terms of meters
        self.shift_range_pixels_lon = shift_range_lon / float(self.meter_per_pixel)  # shift range is in terms of meters

        self.rotation_range = rotation_range  # in terms of degree

        if "train" in self.mode: self.pt_list = args["train_pt_list"]
        elif "valid" in self.mode: self.pt_list = args["val_pt_list"]
        elif "test" in self.mode: self.pt_list = args["test_pt_list"]
        
        self.make_sample_list()
        
    def make_sample_list(self):

        ignore_drive_list = [
            "2011_10_03/2011_10_03_drive_0034_sync/",
            "2011_09_30/2011_09_30_drive_0028_sync/"
        ] # due to download error. broken zip file 
        ignore_file_list = [
            "2011_09_26/2011_09_26_drive_0022_sync/0000000340.png",
            "2011_10_03/2011_10_03_drive_0047_sync/0000000678.png"
        ]
        
        with open(self.pt_list, 'r') as f:
            file_name_list = f.readlines()
        # self.sample_list = [file[:-1] for file in file_name_list]

        self.sample_list = []
        for file_ in file_name_list :
            if file_[:38] in ignore_drive_list: continue
            if file_ in ignore_file_list: continue
            self.sample_list.append(file_[:-1])

        print("[i] {} data loaded, size:{}".format(self.mode, len(self.sample_list)))
        
    def read_data(self, index):
        
        if "train" in self.mode: 
            file_name = self.sample_list[index]
            # day_dir = file_name[:10]
            drive_dir = file_name[:38]
            image_no = file_name[38:]            
            # randomly generate shift
            gt_shift_x = np.random.uniform(-1, 1)  # --> right as positive, parallel to the heading direction
            gt_shift_y = np.random.uniform(-1, 1)  # --> up as positive, vertical to the heading direction
            # randomly generate roation
            theta = np.random.uniform(-1, 1)
            
        else:
            line = self.sample_list[index]
            file_name, gt_shift_x, gt_shift_y, theta = line.split(' ')
            gt_shift_x, gt_shift_y, theta = float(gt_shift_x), float(gt_shift_y), float(theta)
            # day_dir = file_name[:10]
            drive_dir = file_name[:38]
            image_no = file_name[38:]
            
        # =================== read ground image ===================================      
        left_img_name = os.path.join(self.root, drive_dir, "image_02/data", image_no.lower())      
        with Image.open(left_img_name, 'r') as grnd_img:
            grnd_img = grnd_img.convert('RGB')   

        # =================== read satellite map ===================================
        arl_img_name = os.path.join(self.root, "satellite", file_name)
        with Image.open(arl_img_name, 'r') as arl_img:
            arl_img = arl_img.convert('RGB')

        # =================== initialize some required variables ============================
        # oxt: such as 0000000000.txt
        oxts_file_name = os.path.join(self.root, drive_dir, "oxts/data",
                                      image_no.lower().replace('.png', '.txt'))
        with open(oxts_file_name, 'r') as f:
            content = f.readline().split(' ')
            # get heading
            heading = float(content[5])
            heading = torch.from_numpy(np.asarray(heading))            
            
        arl_rot = arl_img.rotate(-heading / np.pi * 180 + 90.)
        arl_align_cam = arl_rot.transform(arl_rot.size, Image.AFFINE,
                                          (1, 0, CameraGPS_shift_left[0] / self.meter_per_pixel,
                                           0, 1, CameraGPS_shift_left[1] / self.meter_per_pixel),
                                          resample=Image.BILINEAR)
        # the homography is defined on: from target pixel to source pixel
        # now north direction is the real vehicle heading direction
        
        return grnd_img, arl_align_cam, gt_shift_x, gt_shift_y, theta
    
    def prep_data(self, grnd_img=None, arl_img=None, gt_shift_x=None, gt_shift_y=None, theta=None):
        
        if grnd_img is not None:        
            grnd_img = self.transform_query(grnd_img)
            
        if arl_img is not None:     
                        
            arl_rand_shift = \
                arl_img.transform(
                    arl_img.size, Image.AFFINE,
                    (1, 0, -gt_shift_x * self.shift_range_pixels_lon,
                    0, 1, -gt_shift_y * self.shift_range_pixels_lat),
                    resample=Image.BILINEAR)
                            
            arl_rand_shift_rand_rot = \
                arl_rand_shift.rotate(theta * self.rotation_range)
         
            arl_img = TF.center_crop(arl_rand_shift_rand_rot, self.arl_img_size[0] * self.arl_zoom_ratio)            
            arl_img = self.transform_reference(arl_img)
        
        return grnd_img, arl_img
    
    def prep_gt(self, gt_shift_x, gt_shift_y, theta):
        
        tgt_y = (self.arl_img_size[0] / 2 + (gt_shift_x * self.shift_range_pixels_lon / self.arl_zoom_ratio)) / self.arl_img_size[0]
        tgt_x = (self.arl_img_size[1] / 2 + (gt_shift_y * self.shift_range_pixels_lat / self.arl_zoom_ratio)) / self.arl_img_size[1]
        
        patch_x = int(tgt_x * (self.arl_img_size[0] / self.down_ratio))
        patch_y = int(tgt_y * (self.arl_img_size[1] / self.down_ratio))
        
        num_query = int((self.arl_img_size[0] / self.down_ratio) * (self.arl_img_size[1] / self.down_ratio))
        
        tgt_class = np.zeros((num_query, 2))
        tgt_class[:, -1] = 1
        tgt_bbox = np.zeros((num_query, 3))
        
        for x_ in range(patch_x - self.spare_pixel, patch_x + self.spare_pixel ):
            for y_ in range(patch_y - self.spare_pixel, patch_y + self.spare_pixel ):
                idx_ = int(x_ * (self.arl_img_size[0] / self.down_ratio) + y_)
                tgt_class[idx_, 0] = 1
                tgt_class[idx_, 1] = 0
                tgt_bbox[idx_, 0] = tgt_x
                tgt_bbox[idx_, 1] = tgt_y                
                tgt_bbox[idx_, 2] = np.deg2rad(theta * self.rotation_range + 180.)
        
        gt = {"labels": torch.tensor(tgt_class),
              "boxes":  torch.tensor(tgt_bbox),
              }    
        return gt
    
    def __getitem__(self, index):

        if self.mode in ["train", "valid"]:
            
            idx = index % len(self.sample_list)            
            grnd_img, arl_img, gt_shift_x, gt_shift_y, theta = self.read_data(idx)        
            img_qry, img_ref = self.prep_data(grnd_img, arl_img, gt_shift_x, gt_shift_y, theta)
            
            gt = self.prep_gt(gt_shift_x, gt_shift_y, theta)
            
            return img_qry, img_ref, gt

        elif self.mode == "valid_ref":                 
            _, arl_img, gt_shift_x, gt_shift_y, theta = self.read_data(index)        
            _, img_ref = self.prep_data(arl_img=arl_img, gt_shift_x=gt_shift_x, gt_shift_y=gt_shift_y, theta=theta)        
            return img_ref, torch.tensor(index), 0

        elif self.mode == "valid_qry":           
            grnd_img, _, _, _, _ = self.read_data(index)            
            img_qry, _ = self.prep_data(grnd_img=grnd_img)
            return img_qry, torch.tensor(index), torch.tensor(index)
        
        else:
            print('not implemented!!')
            raise Exception

    def __len__(self):
        return len(self.sample_list)
