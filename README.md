## STAR-pytorch
Implementation for paper "STAR: A Structure-aware Lightweight Transformer for Real-time Image Enhancement" (ICCV 2021). 

[CVF (pdf)](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhang_STAR_A_Structure-Aware_Lightweight_Transformer_for_Real-Time_Image_Enhancement_ICCV_2021_paper.pdf)

## STAR-DCE
The pytorch implementation of low light enhancement with STAR on Adobe-MIT FiveK dataset. You can find it in STAR-DCE directory. 
Here we adopt the pipleline of Zero-DCE ( [paper](https://li-chongyi.github.io/Proj_Zero-DCE.html) | [code](https://github.com/Li-Chongyi/Zero-DCE) ), just replacing the CNN backbone with STAR.  In Zero-DCE, for each image the network will regress a group of curves, which will then be applied on the source image iteratively. You can find more details in the original repo [Zero-DCE](https://github.com/Li-Chongyi/Zero-DCE). 

### Requirements
- numpy
- einops
- torch
- torchvision
- opencv

### Datesets
We provide download links for Adobe-MIT FiveK datasets we used ( [train](https://drive.google.com/file/d/1skyKKjEIWg0dyxptNLkxrlVddDwr_7vf/view?usp=sharing) | [test](https://drive.google.com/file/d/1C2CXqy-Hu99eehph5GEtmKKUV2a6P-U5/view?usp=sharing) ). 
Please note that we adopt the test set splited by [DeepUPE](https://github.com/dvlab-research/DeepUPE) for fair comparison.

### Training DCE models
To train a original STAR-DCE model, 
```
cd STAR-DCE
python train_dce.py 
  --lowlight_images_path "dir-to-your-training-set" \
  --parallel True \
  --snapshots_folder snapshots/STAR-ori \
  --lr 0.001 \
  --num_epochs 100 \
  --lr_type cos \
  --train_batch_size 32 \
  --model STAR-DCE-Ori \
  --snapshot_iter 10 \
  --num_workers 32 \
```


To train the baseline CNN-based DCE-Net (w\ or w\o Pooling), 
```
cd STAR-DCE
python train_dce.py 
  --lowlight_images_path "dir-to-your-training-set" \
  --parallel True \
  --snapshots_folder snapshots/DCE \
  --lr 0.001 \
  --num_epochs 100 \
  --lr_type cos \
  --train_batch_size 32 \
  --model DCE-Net \
  --snapshot_iter 10 \
  --num_workers 32 \
```
or 
```
cd STAR-DCE
python train_dce.py 
  --lowlight_images_path "dir-to-your-training-set" \
  --parallel True \
  --snapshots_folder snapshots/DCE-Pool \
  --lr 0.001 \
  --num_epochs 100 \
  --lr_type cos \
  --train_batch_size 32 \
  --model DCE-Net-Pool \
  --snapshot_iter 10 \
  --num_workers 32 \
```

### Evaluation of trained models
To evaluated the STAR-DCE model you trained, 
```
cd STAR-DCE
  python test_dce.py \
  --lowlight_images_path  "dir-to-your-test-set" \
  --parallel True \
  --snapshots_folder snapshots_test/STAR-DCE \
  --val_batch_size 1 \
  --pretrain_dir snapshots/STAR-ori/Epoch_best.pth \
  --model STAR-DCE-Ori \
```

To evaluated the DCE-Net model you trained, 
```
cd STAR-DCE
  python test_dce.py \
  --lowlight_images_path  "dir-to-your-test-set" \
  --parallel True \
  --snapshots_folder snapshots_test/DCE \
  --val_batch_size 1 \
  --pretrain_dir snapshots/DCE/Epoch_best.pth \
  --model DCE-Net \
```

### Citation
If this code helps your research, please cite our paper :)

```
@inproceedings{zhang2021star,
  title={STAR: A Structure-Aware Lightweight Transformer for Real-Time Image Enhancement},
  author={Zhang, Zhaoyang and Jiang, Yitong and Jiang, Jun and Wang, Xiaogang and Luo, Ping and Gu, Jinwei},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={4106--4115},
  year={2021}
}
```



