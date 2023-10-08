import torch
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import os
import random
from tqdm import tqdm

from common.utils_loader import input_transform, input_transform_fov

# Same loader from VIGOR, modified for pytorch
# Did you read SliceMatch?
class VIGOR(torch.utils.data.Dataset):
    def __init__(self, mode, args):
        super(VIGOR, self).__init__()

        self.mode = mode
        self.root = args["data_root"]

        if args["fov"] != 0: self.transform_query = input_transform_fov(size=args["grnd_img_size"], fov=args["fov"])
        else: self.transform_query = input_transform(size=args["grnd_img_size"])        
        self.transform_reference = input_transform(size=args["arl_img_size"])
        
        self.same_area = args["same_area"]

        if self.same_area:
            if self.mode == "train":
                self.city_list = ['NewYork', 'Seattle', 'SanFrancisco', 'Chicago']
            else:
                self.city_list = ['NewYork', 'Seattle', 'SanFrancisco', 'Chicago']
        else:
            if self.mode == "train":
                self.city_list = ['NewYork', 'Seattle']
            else:
                self.city_list = ['SanFrancisco', 'Chicago']

        # self.city_list = ["Seattle"] # for test
        
        self.arl_img_size = args["arl_img_size"]
        self.raw_arl_img_size = (640, 640)
        
        self.arl_zoom_ratio = self.raw_arl_img_size[0] / self.arl_img_size[0]
        self.meter_per_pixel_dict = {
            "Chicago": 0.111,
            "NewYork": 0.113,
            "SanFrancisco": 0.118,
            "Seattle": 0.101,
        } # based on SliceMatch
                
        self.make_slice_match_sample_list() 

    def make_slice_match_sample_list(self):
        
        files = ["pano_label_balanced__corrected.txt", "same_area_balanced_train__corrected.txt", "same_area_balanced_test__corrected.txt"]
        
        if self.same_area: 
            if self.mode == "train": file_idx = 1
            else: file_idx = 2
        else: file_idx = 0

        # folder names
        ground_folder_name = "panorama"
        aerial_folder_name = "satellite"
        splits_name = "splits__corrected"
        
        self.arl_fname_to_index_dict = {}
        
        self.sample_list = []
        self.grnd_id_to_arl_id_list = []       
        idx = 0             
        for city in self.city_list:     
            combination_dir = os.path.join(self.root, splits_name, city, files[file_idx])
            ground_img_names = list(sorted(os.listdir(os.path.join(self.root, city, ground_folder_name))))     
            data_dict = get_aerial_and_deltas(combination_dir)
            
            for ground_img_name in tqdm(ground_img_names, desc="[i] dataset load: " + city):  
                if ground_img_name not in data_dict.keys(): continue
                
                data_list = data_dict[ground_img_name] 
                for i in range(len(data_list[:1])):
                    delta = data_list[i][1:]
                    if abs(delta[0])<=self.raw_arl_img_size[0]//2 and abs(delta[1])<=self.raw_arl_img_size[0]//2:                        
                        sample_one = {
                            "grnd_name": os.path.join(os.path.join(city, ground_folder_name), ground_img_name),
                            "arl_name": os.path.join(os.path.join(city, aerial_folder_name), data_list[i][0]),
                            "delta": data_list[i][1:],
                            "is_positive": i == 0,
                            "meter_per_pixel": self.meter_per_pixel_dict[city]
                        }
                        if sample_one["arl_name"] not in self.arl_fname_to_index_dict.keys():
                            self.arl_fname_to_index_dict[sample_one["arl_name"]] = idx
                            idx += 1
                            
                        self.sample_list.append(sample_one)                    
                        self.grnd_id_to_arl_id_list.append(self.arl_fname_to_index_dict[sample_one["arl_name"]])    
        
        # self.sample_list = self.sample_list[:100] # for test 
    
    def read_data(self, index):
                
        grnd_img = Image.open(os.path.join(self.root, self.sample_list[index]["grnd_name"]))
        arl_img = Image.open(os.path.join(self.root, self.sample_list[index]["arl_name"])).convert('RGB')
        
        gt_shift_x = -self.sample_list[index]["delta"][1]
        gt_shift_y = self.sample_list[index]["delta"][0]        
        
        return grnd_img, arl_img, gt_shift_x, gt_shift_y, 0, self.sample_list[index]["meter_per_pixel"]
    
    def prep_data(self, grnd_img=None, arl_img=None):
        
        if grnd_img is not None:        
            grnd_img = self.transform_query(grnd_img)
            
        if arl_img is not None:    
            arl_img = self.transform_reference(arl_img)  
        
        return grnd_img, arl_img
    
    def prep_gt(self, gt_shift_x, gt_shift_y, theta, meter_per_pixel):        
                
        tgt_y = (gt_shift_x / self.arl_zoom_ratio) / self.arl_img_size[1]
        tgt_x = (gt_shift_y / self.arl_zoom_ratio) / self.arl_img_size[0]
        
        tgt_rad = np.deg2rad(theta + 180.)
        tgt_cos = np.cos(tgt_rad)
        tgt_sin = np.sin(tgt_rad)        
                
        target = {
            "boxes": torch.tensor(
                [[tgt_x, tgt_y, tgt_cos, tgt_sin]]
            ),
            "labels": torch.tensor([0]),
            "orig_size": torch.as_tensor([int(self.arl_img_size[0]), int(self.arl_img_size[1])]),   
            "arl_zoom_ratio": torch.tensor([self.arl_zoom_ratio]),     
            "meter_per_pixel": torch.tensor([meter_per_pixel]),   
              }    
        return target
    
    def __getitem__(self, index):
        
        if self.mode in ["train", "valid"]:
            idx = index % len(self.sample_list)            
            grnd_img, arl_img, gt_shift_x, gt_shift_y, theta, meter_per_pixel = self.read_data(idx)        
            img_qry, img_ref = self.prep_data(grnd_img, arl_img)
            gt = self.prep_gt(gt_shift_x, gt_shift_y, theta, meter_per_pixel)
            return img_qry, img_ref, gt
            
        elif self.mode == "valid_ref":            
            _, arl_img, gt_shift_x, gt_shift_y, theta, meter_per_pixel = self.read_data(index)        
            _, img_ref = self.prep_data(arl_img=arl_img)        
            return img_ref, torch.tensor(index), 0
        
        elif self.mode == "valid_qry":
            grnd_img, _, _, _, _, _ = self.read_data(index)            
            img_qry, _ = self.prep_data(grnd_img=grnd_img)
            return img_qry, torch.tensor(index), torch.tensor(self.grnd_id_to_arl_id_list[index])
        else:
            print('not implemented!!')
            raise Exception

    def __len__(self):
        return len(self.sample_list)

def get_aerial_and_deltas(combination_dir):
    data_dict = {}
    with open(combination_dir, 'r') as file:
        for line in file.readlines():
            data = line.split(' ')
            data_list = []
            for idx in range(4):
                data_list.append((data[3*idx+1], float(data[3*idx+2]), float(data[3*idx+3])))                    
            data_dict[data[0]] = data_list
    return data_dict