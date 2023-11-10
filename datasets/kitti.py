import torch
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
        
        self.transform_query = input_transform(size=args["grd_img_size"])        
        self.transform_reference = input_transform(size=args["arl_img_size"])
        
        self.arl_img_size = args["arl_img_size"]
        self.grd_img_size = args["grd_img_size"] # 256, 1024 
        
        shift_range_lat = float(args["shift_range_lat"])
        shift_range_lon = float(args["shift_range_lon"])
        rotation_range = float(args["rotation_range"])

        self.meter_per_pixel = get_meter_per_pixel(scale=1)
        self.shift_range_meters_lat = shift_range_lat  # in terms of meters
        self.shift_range_meters_lon = shift_range_lon  # in terms of meters
        self.shift_range_pixels_lat = shift_range_lat / float(self.meter_per_pixel)  # shift range is in terms of meters
        self.shift_range_pixels_lon = shift_range_lon / float(self.meter_per_pixel)  # shift range is in terms of meters

        self.rotation_range = rotation_range  # in terms of degree
        
        if "train" in self.mode: self.pt_list = args["train_pt_list"]
        elif "valid_cross" in self.mode: self.pt_list = args["val_cross_pt_list"]
        elif "valid_same" in self.mode: self.pt_list = args["val_same_pt_list"]
        elif "test" in self.mode: self.pt_list = args["test_pt_list"]
        
        self.make_sample_list()
        
    def make_sample_list(self):

        ignore_drive_list = [
        ] # due to download error. broken zip file 
        ignore_file_list = [
            # "2011_10_03/2011_10_03_drive_0034_sync/",
            # "2011_09_30/2011_09_30_drive_0028_sync/"
        ] # due to download error. broken zip file 
        
        with open(self.pt_list, 'r') as f:
            file_name_list = f.readlines()
        # self.sample_list = [file[:-1] for file in file_name_list]

        self.sample_list = []
        for file_ in file_name_list :
            file_ = file_[:-1]
            if file_[:37] in ignore_drive_list: continue
            if file_[:52] in ignore_file_list: continue
            self.sample_list.append(file_)
        FileNotFoundError("[i] {} data loaded, size:{}".format(self.mode, len(self.sample_list)))
    
    def __getitem__(self, index):
        if self.mode == "train" or self.mode == "valid_same" or self.mode == "valid_cross":            
            idx = index % len(self.sample_list)    
            
            if "train" in self.pt_list: 
                file_name = self.sample_list[idx].split(' ')[0]
                # day_dir = file_name[:10]
                drive_dir = file_name[:38]
                image_no = file_name[38:]            
                # randomly generate shift
                gt_shift_x = np.random.uniform(-1, 1)  # --> right as positive, parallel to the heading direction
                gt_shift_y = np.random.uniform(-1, 1)  # --> up as positive, vertical to the heading direction
                # randomly generate roation
                theta = np.random.uniform(-1, 1)
            else:
                line = self.sample_list[idx]
                file_name, gt_shift_x, gt_shift_y, theta = line.split(' ')
                gt_shift_x, gt_shift_y, theta = float(gt_shift_x), float(gt_shift_y), float(theta)
                # day_dir = file_name[:10]
                drive_dir = file_name[:38]
                image_no = file_name[38:]
                
                # if self.mode == "valid_same" or self.mode == "valid_cross":
                #     print("kitti valid theta ", theta)
                
            # =================== read ground image ===================================      
            left_img_name = os.path.join(self.root, "raw", drive_dir, "image_02/data", image_no.lower())   
            try: grd_img = Image.open(left_img_name, 'r'); grd_img = grd_img.convert('RGB')   
            except: FileNotFoundError(f'{left_img_name} not exists')

            # =================== read satellite map ==================================
            arl_img_name = os.path.join(self.root, "satellite", file_name)
            try: arl_img = Image.open(arl_img_name, 'r'); arl_img = arl_img.convert('RGB')   
            except: FileNotFoundError(f'{arl_img_name} not exists')

            # =================== initialize some required variables ==================
            # oxt: such as 0000000000.txt
            oxts_file_name = os.path.join(self.root, "raw", drive_dir, "oxts/data",
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
            arl_img = arl_align_cam
            # the homography is defined on: from target pixel to source pixel
            # now north direction is the real vehicle heading direction
            
            # =================== add random translation & rotation ===================            
            grd_img = self.transform_query(grd_img)    
            
            arl_rand_rot = \
                arl_img.rotate(theta * self.rotation_range)  
            
            arl_rand_rot_rand_shift = \
                arl_rand_rot.transform(
                    arl_img.size, Image.AFFINE,
                    (1, 0, -gt_shift_x * self.shift_range_pixels_lon,
                    0, 1, -gt_shift_y * self.shift_range_pixels_lat),
                    resample=Image.BILINEAR)
         
            arl_img = TF.center_crop(arl_rand_rot_rand_shift, self.arl_img_size[0])            
            arl_img = self.transform_reference(arl_img)  
            
            # =================== make target dict ====================================
            tgt_y = (gt_shift_x * self.shift_range_pixels_lon) / self.arl_img_size[1]
            tgt_x = (gt_shift_y * self.shift_range_pixels_lat) / self.arl_img_size[0]
            
            tgt_rad = np.deg2rad(theta * self.rotation_range + 180.)
            tgt_cos = np.cos(tgt_rad)
            tgt_sin = np.sin(tgt_rad)
                    
            target = {
                "boxes": torch.tensor(
                    [[tgt_x, tgt_y, tgt_cos, tgt_sin]]
                ),
                "labels": torch.tensor([0]),
                "orig_size": torch.as_tensor([int(self.arl_img_size[0]), int(self.arl_img_size[1])]),      
                "meter_per_pixel": torch.tensor([self.meter_per_pixel]),  
            } 
                    
            return grd_img, arl_img, target

        elif self.mode == "valid_same_ref" or self.mode == "valid_cross_ref":            
            line = self.sample_list[index]
            file_name, gt_shift_x, gt_shift_y, theta = line.split(' ')
            gt_shift_x, gt_shift_y, theta = float(gt_shift_x), float(gt_shift_y), float(theta)
            # day_dir = file_name[:10]
            drive_dir = file_name[:38]
            image_no = file_name[38:]

            # =================== read satellite map ==================================
            arl_img_name = os.path.join(self.root, "satellite", file_name)
            try: arl_img = Image.open(arl_img_name, 'r'); arl_img = arl_img.convert('RGB')   
            except: FileNotFoundError(f'{arl_img_name} not exists')

            # =================== initialize some required variables ==================
            # oxt: such as 0000000000.txt
            oxts_file_name = os.path.join(self.root, "raw", drive_dir, "oxts/data",
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
            arl_img = arl_align_cam
            # the homography is defined on: from target pixel to source pixel
            # now north direction is the real vehicle heading direction 
            
            # =================== add random translation & rotation ===================  
            arl_rand_rot = \
                arl_img.rotate(theta * self.rotation_range)  
            
            arl_rand_rot_rand_shift = \
                arl_rand_rot.transform(
                    arl_img.size, Image.AFFINE,
                    (1, 0, -gt_shift_x * self.shift_range_pixels_lon,
                    0, 1, -gt_shift_y * self.shift_range_pixels_lat),
                    resample=Image.BILINEAR)
         
            arl_img = TF.center_crop(arl_rand_rot_rand_shift, self.arl_img_size[0])            
            arl_img = self.transform_reference(arl_img)   
            
            return arl_img, torch.tensor(index), 0

        elif self.mode == "valid_same_qry" or self.mode == "valid_cross_qry":   
            
            line = self.sample_list[index]
            file_name, gt_shift_x, gt_shift_y, theta = line.split(' ')
            gt_shift_x, gt_shift_y, theta = float(gt_shift_x), float(gt_shift_y), float(theta)
            # day_dir = file_name[:10]
            drive_dir = file_name[:38]
            image_no = file_name[38:]
            
            # =================== read ground image ===================================      
            left_img_name = os.path.join(self.root, "raw", drive_dir, "image_02/data", image_no.lower())   
            try: grd_img = Image.open(left_img_name, 'r'); grd_img = grd_img.convert('RGB')   
            except: FileNotFoundError(f'{left_img_name} not exists')        
                    
            grd_img = self.transform_query(grd_img)
            
            return grd_img, torch.tensor(index), torch.tensor(index)
        else:
            NotImplementedError()

    def __len__(self):
        return len(self.sample_list)
