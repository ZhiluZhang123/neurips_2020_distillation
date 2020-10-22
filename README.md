# Self-Distillation as Instance-Specific Label Smoothing

## Overview
This repository contains implementation for our paper "Self-Distillation as Instance-Specific Label Smoothing".

Zhang, Zhilu, and Mert R. Sabuncu. "Self-Distillation as Instance-Specific Label Smoothing." arXiv preprint arXiv:2006.05065 (2020). https://arxiv.org/abs/2006.05065

## Content
`train_sitillation.py` contains the script for training and evaluating multi-generational self-distillation. Trained models are saved in the `saved_models` directory.

`train_smoothing.py` contains the script for training and evaluating with various kinds of smoothing techniques, including regular label smoothing, the proposed Beta smoothing and also regular cross entropy with entropy regularization. Note that the parameter `--beta` has different use when different loss is used. For regular label smoothing, `--beta` sets the `epsilon` parameter in label smoothing; for Beta smoothing, `--beta` is the parameter for Beta distribution (see paper for more details).

`ece.py` contains the code for expected calibration error.

`loss.py` contains various types of loss functions used for training. 

`tools.py` contains functions for evaluating a trained model.

## Citation 
Please cite our paper if you find this useful, thank you! 

```
@article{zhang2020self,
  title={Self-Distillation as Instance-Specific Label Smoothing},
  author={Zhang, Zhilu and Sabuncu, Mert R},
  journal={arXiv preprint arXiv:2006.05065},
  year={2020}
}
```

