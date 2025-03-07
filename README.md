# Co-MTP: A Cooperative Trajectory Prediction Framework with Multi-Temporal Fusion for Autonomous Driving
![The overall architecture of Co-MTP.](static/images/overview.jpg)

## Introduction
Co-MTP is a general cooperative trajectory prediction framework with multi-temporal fusion for autonomous driving, which leverages the V2X system to fully capture the interaction among agents in both history and future domains to benefit the planning.

We evaluate the Co-MTP framework on the real-world dataset [V2X-Seq](https://github.com/AIR-THU/DAIR-V2X-Seq), and the results show that Co-MTP achieves **SOTA** performance and that both history and future fusion can greatly benefit prediction.

## Installation
We suggest using exactly the same environment as provided to avoid any issues:

```bash
conda create -n co-mtp python=3.8
conda activate co-mtp

pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install tensorflow tensorboard

wget https://anaconda.org/dglteam/dgl-cuda11.3/0.7.2/download/linux-64/dgl-cuda11.3-0.7.2-py38_0.tar.bz2
conda install --use-local dgl-cuda11.3-0.7.2-py38_0.tar.bz2 -y
conda install protobuf=3.20 -y

pip install -r requirements.txt
```

## Dataset Preparation
1. Download the dataset V2X-Seq-TFD through the dataset link provided above and save the data into the {Data_root} directory './dataset/V2X-Seq-TFD'.

2. Merge multiple maps into one map
```bash
python 
```


## Training & Evaluation
