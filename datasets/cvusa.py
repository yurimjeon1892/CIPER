import torch
import os
from PIL import Image

from common.utils_loader import input_transform, input_transform_fov

# pytorch version of CVUSA loader
class CVUSA(torch.utils.data.Dataset):
    def __init__(self, mode, args): # CV-dataset
        super(CVUSA, self).__init__()
        
        self.mode = mode
        self.root = args["data_root"]
        
        if args["fov"] != 0: self.transform_query = input_transform_fov(size=args["grd_img_size"], fov=args["fov"])
        else: self.transform_query = input_transform(size=args["grd_img_size"])
        self.transform_reference = input_transform(size=args["arl_img_size"])
                        
        if "train" in self.mode: self.pt_list = args["train_pt_list"]
        elif "valid" in self.mode: self.pt_list = args["val_pt_list"]
        elif "test" in self.mode: self.pt_list = args["test_pt_list"]
        
        self.make_sample_list()
        
    def make_sample_list(self):

        self.sample_list = []
        self.id_idx_list = []
        with open(self.pt_list, 'r') as file:
            idx = 0
            for line in file:
                data = line.split(',')
                pano_id = (data[0].split('/')[-1]).split('.')[0]
                # satellite filename, streetview filename, pano_id
                self.sample_list.append([data[0], data[1], pano_id])
                self.id_idx_list.append(idx)
                idx += 1
        
        print("[i] {} data loaded, size:{}".format(self.mode, len(self.sample_list)))
        
    def read_data(self, index):                
        grd_img = Image.open(os.path.join(self.root, self.sample_list[index][1])).convert('RGB')
        arl_img = Image.open(os.path.join(self.root, self.sample_list[index][0])).convert('RGB')        
        return grd_img, arl_img
    
    def prep_data(self, grd_img=None, arl_img=None):                
        if grd_img is not None:        
            grd_img = self.transform_query(grd_img)            
        if arl_img is not None:                 
            arl_img = self.transform_reference(arl_img)        
        return grd_img, arl_img

    def __getitem__(self, index):
        
        if "train" in self.mode:            
            idx = index % len(self.sample_list)
            grd_img, arl_img = self.read_data(idx)        
            img_qry, img_ref = self.prep_data(grd_img, arl_img)                       
            return img_qry, img_ref, torch.tensor(idx)

        elif "valid_ref" in self.mode:            
            _, arl_img = self.read_data(index)        
            _, img_ref = self.prep_data(arl_img=arl_img)        
            return img_ref, torch.tensor(index), 0

        elif "valid_qry" in self.mode:            
            # grd_img, _ = self.read_data(index)            
            # img_qry, _ = self.prep_data(grd_img=grd_img)
            # return img_qry, torch.tensor(index), torch.tensor(index)
            return 0, torch.tensor(index), torch.tensor(index)
        
        else:
            print('not implemented!!')
            raise Exception

    def __len__(self):
        return len(self.sample_list)