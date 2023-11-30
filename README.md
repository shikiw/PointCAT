# PointCAT
This repository provides the official PyTorch implementation of the following paper: 
> [**PointCAT: Contrastive Adversarial Training for Robust Point Cloud Recognition**](https://arxiv.org/abs/2209.07788) <br>
> [Qidong Huang](https://shikiw.github.io/)<sup>1</sup>, 
> [Xiaoyi Dong](https://scholar.google.com/citations?user=FscToE0AAAAJ&hl=en)<sup>1</sup>, 
> [Dongdong Chen](https://www.dongdongchen.bid/)<sup>2</sup>, 
> [Hang Zhou](http://www.sfu.ca/~hza162/)<sup>3</sup>, 
> [Weiming Zhang](http://staff.ustc.edu.cn/~zhangwm/index.html)<sup>1</sup>, 
> Kui Zhang<sup>1</sup>, 
> [Gang Hua](https://www.ganghua.org/)<sup>4</sup>, 
> [Nenghai Yu](https://scholar.google.com/citations?user=7620QAMAAAAJ&hl=en)<sup>1</sup> <br>
> <sup>1</sup>University of Science and Technology of China, <sup>2</sup>Microsoft Cloud AI, <sup>3</sup>Simon Fraser University, <sup>4</sup>Wormpex AI Research <br>
>

## Environment Setup
This code is tested with Python3.7 and CUDA = 10.3, to setup a conda environment, please use the following instructions:
```
conda env create -f environment.yaml
conda activate pointcat
```

## Preparation
Download the aligned [ModelNet40](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip) dataset and [ShapeNetPart](https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_segmentation_benchmark_v0.zip) dataset in their point cloud format and unzip them into your own dataset path.
You can also run the bash script:
```
sh download.sh
```
the datasets will be downloaded at ```./data``` by default.

Download the [pretrained models](https://drive.google.com/drive/folders/14xgqEjBmTkQK7wnz2kCIz9LaHNPGN1Pk?usp=sharing) we provided for attack evaluation and unzip them at ```./checkpoint```. The available models include 
[PointNet](https://github.com/charlesq34/pointnet), 
[PointNet++](https://github.com/charlesq34/pointnet2), 
[DGCNN](https://github.com/WangYueFt/dgcnn) and 
[CurveNet](https://github.com/tiangexiang/CurveNet).

## Evaluation
You can directly evaluate our released pretrained models. For example, please run the following command for PointNet on ModelNet40:
```
python tester.py \
--data_path /PATH/TO/YOUR/DATASET/ \
--dataset ModelNet40 \
--defended_model pointnet_cls \
--batch_size 16 \
--mode test_normal \
--checkpoint_dir ./checkpoints/pointnet_pointcat_mn.pth
```

## Training
To implement training for PointNet, please run the following command:
```
python trainer.py \
--experiment_dir pn_test \
--data_path /PATH/TO/YOUR/DATASET/ \
--dataset ModelNet40 \
--defended_model pointnet_cls \
--eps 0.04 \
--alpha 8. \
--beta 0.5 \
--use_cosine_similarity  \
--inner_loop_nums 4 \
--batch_size 64 \
--init_search_iters 500 \
--update_search_iters 10 \
--lr_fp 0.001 \
--use_multi_gpu
```

## Citation
If you find this work useful for your research, please cite our [paper](https://arxiv.org/abs/2209.07788):
```
@article{huang2022pointcat,
  title={PointCAT: Contrastive Adversarial Training for Robust Point Cloud Recognition},
  author={Huang, Qidong and Dong, Xiaoyi and Chen, Dongdong and Zhou, Hang and Zhang, Weiming and Zhang, Kui and Hua, Gang and Yu, Nenghai},
  journal={arXiv preprint arXiv:2209.07788},
  year={2022}
}
```

## License
The code is released under MIT License (see LICENSE file for details).
