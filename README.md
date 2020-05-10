# PointPWC
This is the code for [PointPWC-Net](https://arxiv.org/abs/1911.12408), a deep coarse-to-fine network designed for 3D scene flow estimation from 3D point clouds.

## Prerequisities
Our model is trained and tested under:
* Python 3.6.9
* NVIDIA GPU + CUDA CuDNN
* PyTorch (torch == 1.5)
* scipy
* tqdm
* sklearn
* numba
* cffi

Compile the furthest point sampling, grouping and gathering operation for PyTorch. We use the operation from this [repo](https://github.com/sshaoshuai/Pointnet2.PyTorch).

```shell
cd pointnet2
python setup.py install
cd ../
```

## Data preprocess

For fair comparison with previous methods, we adopt the preprocessing steps in [HPLFlowNet](https://web.cs.ucdavis.edu/~yjlee/projects/cvpr2019-HPLFlowNet.pdf). Please refer to [repo](https://github.com/laoreja/HPLFlowNet). We alos copy the preprocessing instructions here for your reference.

* FlyingThings3D:
Download and unzip the "Disparity", "Disparity Occlusions", "Disparity change", "Optical flow", "Flow Occlusions" for DispNet/FlowNet2.0 dataset subsets from the [FlyingThings3D website](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html) (we used the paths from [this file](https://lmb.informatik.uni-freiburg.de/data/FlyingThings3D_subset/FlyingThings3D_subset_all_download_paths.txt), now they added torrent downloads)
. They will be upzipped into the same directory, `RAW_DATA_PATH`. Then run the following script for 3D reconstruction:

```bash
python3 data_preprocess/process_flyingthings3d_subset.py --raw_data_path RAW_DATA_PATH --save_path SAVE_PATH/FlyingThings3D_subset_processed_35m --only_save_near_pts
```

* KITTI Scene Flow 2015
Download and unzip [KITTI Scene Flow Evaluation 2015](http://www.cvlibs.net/download.php?file=data_scene_flow.zip) to directory `RAW_DATA_PATH`.
Run the following script for 3D reconstruction:

```bash
python3 data_preprocess/process_kitti.py RAW_DATA_PATH SAVE_PATH/KITTI_processed_occ_final
```

## Get started

Here are some demo results:

<img src="./images/FlyingThings3D.gif" width=50%>

<img src="./images/Kitti.gif" width=50%>

### Train
Set `data_root` in the configuration file to `SAVE_PATH` in the data preprocess section. Then run
```bash
python3 train.py config_train.yaml
```

### Evaluate
Set `data_root` in the configuration file to `SAVE_PATH` in the data preprocess section. Then run
```bash
python3 evaluate.py config_evaluate.yaml
```

We upload one pretrained model in ```pretrain_weights```.

## Citation

If you use this code for your research, please cite our paper.

```
@article{wu2019pointpwc,
  title={PointPWC-Net: A Coarse-to-Fine Network for Supervised and Self-Supervised Scene Flow Estimation on 3D Point Clouds},
  author={Wu, Wenxuan and Wang, Zhiyuan and Li, Zhuwen and Liu, Wei and Fuxin, Li},
  journal={arXiv preprint arXiv:1911.12408},
  year={2019}
}
```

## Acknowledgement

We thank [repo](https://github.com/sshaoshuai/Pointnet2.PyTorch) and [repo](https://github.com/laoreja/HPLFlowNet) for subsampling, grouping and data preprocessing related functions.


