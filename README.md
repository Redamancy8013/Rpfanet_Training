# Rpfanet_Training

<div align="center">
  <img src="https://github.com/Redamancy8013/Rpfanet_Training/blob/main/Astyx.jpg">
</div>

This project is used to train the model of the object detection network with point clouds only. It is based on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) framework which is a  a clear, simple, self-contained open source project for LiDAR-based 3D object detection. Till now, the [Astyx](https://github.com/under-the-radar/radar_dataset_astyx) dataset has been successfully handled and trained.

## 1.Notice

After downloading the dataset, put them into the proper directory.

In this project, eg. the `Astyx` dataset is put into `/home/ez/project/rpfanet/data/Astyx/`. In the directory `/Astyx`, there are `3` folders named training, testing, ImageSets.

## 2.Environment

`conda env create -f environment.yml`

`conda activate OpenPCDet`

##### !Attention!

* Install the SparseConv library, we use the implementation from [`[spconv]`](https://github.com/traveller59/spconv). 
    * If you use PyTorch 1.1, then make sure you install the `spconv v1.0` with ([commit 8da6f96](https://github.com/traveller59/spconv/tree/8da6f967fb9a054d8870c3515b1b44eca2103634)) instead of the latest one.
    * If you use PyTorch 1.3+, then you need to install the `spconv v1.2`. As mentioned by the author of [`spconv`](https://github.com/traveller59/spconv), you need to use their docker if you use PyTorch 1.4+. 

## 3.Generate dataloader

```
python -m pcdet.datasets.astyx.astyx_dataset create_astyx_infos tools/cfgs/dataset_configs/astyx_dataset_radar.yaml

python3 -m pcdet.datasets.astyx.ttttt create_astyx_infos tools/cfgs/dataset_configs/astyx_dataset_radar.yaml
```

## 4.Training

```
cd tools/

python train.py --cfg_file cfgs/astyx_models/pointpillar.yaml --tcp_port 25851 --extra_tag yourmodelname
```

###### or

```shell
cd tools/

CUDA_VISIBLE_DEVICES=1 python train.py --cfg_file cfgs/astyx_models/pointpillar.yaml --tcp_port 25851 --extra_tag yourmodelname
```

## 5.Testing

```
cd tools/

python test.py --cfg_file cfgs/astyx_models/pointpillar.yaml --batch_size 4 --ckpt ../output/astyx_models/pointpillar/rpfanet_313/ckpt/checkpoint_epoch_80.pth
```

###### or

```shell
cd tools/

python test.py --cfg_file cfgs/astyx_models/pointpillar.yaml --batch_size 4 --ckpt ##astyx_models/pointpillar/debug/ckpt/checkpoint_epoch_80.pth
```

## 6.Visualization

'''
python3 demo.py --ckpt ../output/astyx_models/pointpillar/rpfanet_313/ckpt/checkpoint_epoch_80.pth
'''

## 7.Main directory struction

```
├─data
│  └─Astyx
│      ├─ImageSets
│      ├─training
│      └─testing
├─docker
├─docs
├─pcdet
│  ├─datasets
│  │  ├─astyx
│  │  ├─augmentor
│  │  ├─kitti
│  │  │  └─kitti_object_eval_python
│  │  ├─nuscenes
│  │  └─processor
│  ├─models
│  │  ├─backbones_2d
│  │  │  └─map_to_bev
│  │  ├─backbones_3d
│  │  │  ├─pfe
│  │  │  └─vfe
│  │  ├─dense_heads
│  │  │  └─target_assigner
│  │  ├─detectors
│  │  ├─model_utils
│  │  └─roi_heads
│  │      └─target_assigner
│  ├─ops
│  │  ├─iou3d_nms
│  │  │  └─src
│  │  ├─pointnet2
│  │  │  ├─pointnet2_batch
│  │  │  │  └─src
│  │  │  └─pointnet2_stack
│  │  │      └─src
│  │  ├─roiaware_pool3d
│  │  │  └─src
│  │  └─roipoint_pool3d
│  │      └─src
│  └─utils
└─tools
    ├─cfgs
    │  ├─astyx_models
    │  ├─dataset_configs
    │  ├─kitti_models
    │  └─nuscenes_models
    ├─eval_utils
    ├─scripts
    ├─train_utils
    │  └─optimization
    └─visual_utils
```

## 8.More

This project was accomplished on June 1, 2024 and was first upload to the github on March 11, 2025.

Contact Email: 2110539202@qq.com
