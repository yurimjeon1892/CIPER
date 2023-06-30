# CITRUS :tangerine:

Cross-view Image geo-localization TRansformer 

This is what you came for!!

## Environment

### Step 1: Requirements
* CUDA 11.3
* cuDNN 8
* Ubuntu 20.04

### Step 2: Create conda environment
```
conda create -n citrus python=3.8
conda activate citrus
pip install -r requirements.txt
```

## Data


## Train 
Set `data_root` and `ckpt_root` in the `train.yaml` file and run:
```
python main.py configs/train.yaml
```

## Test
Set `resume_path` in the `test.yaml` file and run:
```
python test.py configs/test.yaml
```
