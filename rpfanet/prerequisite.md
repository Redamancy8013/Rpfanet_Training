conda activate OpenPCDet

# 数据预处理

python3 -m pcdet.datasets.astyx.astyx_dataset create_astyx_infos tools/cfgs/dataset_configs/astyx_dataset_radar.yaml

python3 -m pcdet.datasets.astyx.ttttt create_astyx_infos tools/cfgs/dataset_configs/astyx_dataset_radar.yaml

python3 -m pcdet.datasets.astyx.vod_dataset create_vod_infos tools/cfgs/dataset_configs/astyx_dataset_radar.yaml


python3 -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml




python3 train.py --cfg_file cfgs/kitti_models/pointpillar.yaml --tcp_port 25851 --extra_tag kitti425

python3 test.py --cfg_file cfgs/astyx_models/pointpillar.yaml --batch_size 4 --ckpt ../output/astyx_models/pointpillar/rpfanet_313/ckpt/checkpoint_epoch_80.pth


python3 test.py --cfg_file cfgs/kitti_models/pointpillar.yaml --batch_size 4 --ckpt ../output/kitti_models/pointpillar/kitti415/ckpt/checkpoint_epoch_5.pth

python3 demo.py --ckpt ../output/astyx_models/pointpillar/rpfanet_313/ckpt/checkpoint_epoch_80.pth