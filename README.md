# CIPER :round_pushpin:

**C**ross-view **I**mage-retrieval and **P**ose-estimation transform**ER** 

## Environment

### Step 1: Requirements
* CUDA 11.3
* cuDNN 8
* Ubuntu 20.04

### Step 2: Create conda environment
```
conda create -n ciper python=3.8
conda activate ciper
pip install -r requirements.txt
```

## Data

### KITTI

Download KITTI [street-view images](https://www.cvlibs.net/datasets/kitti/raw_data.php) and [satellite images](https://github.com/shiyujiao/HighlyAccurate). The folder structure is as follows:
```
KITTI
├── raw
|   ├── 2011_09_26
|   |   ├── 2011_09_26_drive_0001_sync
|   |   |   ├── image_00
|   |   |   ├── image_01
|   |   |   ├── image_02
|   |   |   ├── image_03
|   |   |   └── oxts    
|   |   ├── 2011_09_26_drive_0002_sync    
|   |   └── ..
|   ├── 2011_09_28
|   └── ..
└── satellite
    ├── 2011_09_26
    |   ├── 2011_09_26_drive_0001_sync
    |   ├── 2011_09_26_drive_0002_sync
    |   └── ..
    ├── 2011_09_28
    └── ..

```

## Run

### Train 
Set `data_root` and `ckpt_root` in the `train.yaml` file and run:
```
python main.py configs/train.yaml
```

### Test
Set `resume_path` in the `test.yaml` file and run:
```
python main.py configs/test.yaml
```

## Acknowledgements
This project is not possible without the following awesome open-source codebases:
* [TransGeo: Transformer Is All You Need for Cross-view Image Geo-localization](https://github.com/Jeff-Zilence/TransGeo2022)
* [Beyond Cross-view Image Retrieval: Highly Accurate Vehicle Localization Using Satellite Image](https://github.com/shiyujiao/HighlyAccurate)
* [DETR: End-to-End Object Detection with Transformers](https://github.com/facebookresearch/detr)
* [segment-anything](https://github.com/facebookresearch/segment-anything)
