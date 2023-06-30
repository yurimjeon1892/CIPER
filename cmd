python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --coco_path /storage2/public/coco2017 


python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --data_root /storage2/public/kitti 

