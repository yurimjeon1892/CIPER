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

        self.city_list = ["Seattle"]
        
        self.arl_img_size = args["arl_img_size"]
        self.raw_arl_img_size = (640, 640)
        
        self.down_ratio = float(args["down_ratio"])
        self.spare_pixel = int(args["spare_pixel"])
        
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
            ground_img_names = ground_img_names[:100]
            
            for ground_img_name in tqdm(ground_img_names):   
                data_list = get_aerial_and_deltas(ground_img_name, combination_dir)
                for i in range(len(data_list[:1])):
                    sample_one = {
                        "grnd_name": os.path.join(os.path.join(city, ground_folder_name), ground_img_name),
                        "arl_name": os.path.join(os.path.join(city, aerial_folder_name), data_list[i][0]),
                        "delta": data_list[i][1:],
                        "is_positive": i == 0,
                    }
                    if sample_one["arl_name"] not in self.arl_fname_to_index_dict.keys():
                        self.arl_fname_to_index_dict[sample_one["arl_name"]] = idx
                        idx += 1
                        
                    self.sample_list.append(sample_one)                    
                    self.grnd_id_to_arl_id_list.append(self.arl_fname_to_index_dict[sample_one["arl_name"]])    
        
        self.sample_list = self.sample_list[:100]                
                    
        return
    
    def read_data(self, index):
                
        grnd_img = Image.open(os.path.join(self.root, self.sample_list[index]["grnd_name"]))
        arl_img = Image.open(os.path.join(self.root, self.sample_list[index]["arl_name"])).convert('RGB')
        
        gt_shift_x = -self.sample_list[index]["delta"][1]
        gt_shift_y = self.sample_list[index]["delta"][0]        
        
        return grnd_img, arl_img, gt_shift_x, gt_shift_y, 0
    
    def prep_data(self, grnd_img=None, arl_img=None):
        
        if grnd_img is not None:        
            grnd_img = self.transform_query(grnd_img)
            
        if arl_img is not None:    
            arl_img = self.transform_reference(arl_img)        
        
        return grnd_img, arl_img
    
    def prep_gt(self, gt_shift_x, gt_shift_y, theta):
        
        # print(gt_shift_x, gt_shift_y)
                
        tgt_y = (self.raw_arl_img_size[0] / 2 + gt_shift_x) / self.raw_arl_img_size[0]
        tgt_x = (self.raw_arl_img_size[1] / 2 + gt_shift_y) / self.raw_arl_img_size[1]
                
        patch_x = int(tgt_x * (self.arl_img_size[0] / self.down_ratio))
        patch_y = int(tgt_y * (self.arl_img_size[1] / self.down_ratio))
        
        num_query = int((self.arl_img_size[0] / self.down_ratio) * (self.arl_img_size[1] / self.down_ratio))
        
        tgt_class = np.zeros((num_query, 2))
        tgt_class[:, -1] = 1
        tgt_bbox = np.zeros((num_query, 3))
        
        for x_ in range(patch_x - self.spare_pixel, patch_x + self.spare_pixel ):
            for y_ in range(patch_y - self.spare_pixel, patch_y + self.spare_pixel ):
                idx_ = int(x_ * (self.arl_img_size[0] / self.down_ratio) + y_)
                if idx_ < 0 or idx_ >= num_query: continue
                tgt_class[idx_, 0] = 1
                tgt_class[idx_, 1] = 0
                tgt_bbox[idx_, 0] = tgt_x
                tgt_bbox[idx_, 1] = tgt_y                
                tgt_bbox[idx_, 2] = np.pi
        
        gt = {"labels": torch.tensor(tgt_class),
              "boxes":  torch.tensor(tgt_bbox)
              }    
        return gt
    
    def __getitem__(self, index):
        
        if self.mode in ["train", "valid"]:
            idx = index % len(self.sample_list)            
            grnd_img, arl_img, gt_shift_x, gt_shift_y, theta = self.read_data(idx)        
            img_qry, img_ref = self.prep_data(grnd_img, arl_img)
            gt = self.prep_gt(gt_shift_x, gt_shift_y, theta)
            return img_qry, img_ref, gt
            
        elif self.mode == "valid_ref":            
            _, arl_img, gt_shift_x, gt_shift_y, theta = self.read_data(index)        
            _, img_ref = self.prep_data(arl_img=arl_img)        
            return img_ref, torch.tensor(index), 0
        
        elif self.mode == "valid_qry":
            grnd_img, _, _, _, _ = self.read_data(index)            
            img_qry, _ = self.prep_data(grnd_img=grnd_img)
            return img_qry, torch.tensor(index), torch.tensor(self.grnd_id_to_arl_id_list[index])
        else:
            print('not implemented!!')
            raise Exception

    def __len__(self):
        return len(self.sample_list)

def get_aerial_and_deltas(ground_img_name, combination_dir):
    data_list = []
    with open(combination_dir, 'r') as file:
        for line in file.readlines():
            data = line.split(' ')
            if data[0]==ground_img_name:
                for idx in range(4):
                    data_list.append((data[3*idx+1], float(data[3*idx+2]), float(data[3*idx+3])))
                break
    return data_list