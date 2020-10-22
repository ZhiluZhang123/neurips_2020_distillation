# Self-Distillation as Instance-Specific Label Smoothing

## Overview
This repository contains implementation for our paper "Self-Distillation as Instance-Specific Label Smoothing".

Zhang, Zhilu, and Mert R. Sabuncu. "Self-Distillation as Instance-Specific Label Smoothing." arXiv preprint arXiv:2006.05065 (2020). https://arxiv.org/abs/2006.05065

## Content
`train_sitillation.py` contains the script for training multi-generational self-distillation. 

`train_smoothing.py` contains the script for training with various kinds of smoothing techniques, including regular label smoothing, the proposed Beta smoothing and also regular cross entropy with entropy regularization.


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

