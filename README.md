# Automatic Synthesis of Diverse Weak Supervision Sources for Behavior Analysis ([AutoSWAP](https://arxiv.org/abs/2111.15186))
Albert Tseng, Jennifer J. Sun, & Yisong Yue. CVPR 2022.

## Overview

Labeling lots of data is difficult and costly. Our framework, AutoSWAP, reduces domain expert effort by combining weak supervision, program synthesis, and deep learning, to improve data efficiency. Our experiments in behavioral neuroscience and sports analytics settings show that AutoSWAP matches or outperforms baseline methods using a fraction of the data. Please see the full paper at https://arxiv.org/abs/2111.15186 for more details.

## Code details

This code is the "cleaned up" version of the main experiment code. It should work out of the box but there may be some uncaught errors. It is written to the standard of typical academic code, and in no way reflects on the code quality of any of the authors' current or prior employers. Feel free to email atseng@caltech.edu if you have any issues running the code.

- `train*.py` contains various training scripts for running experiments
- `lib/` contains various helper modules and functions for the training scripts
- `external/NEAR/` contains our modified version of the NEAR[1] program synthesizer
- `external/NEAR/algorithms/diversity` contains our implementation of the diversity loss integrated into NEAR

[1] Shah et al., Learning Differentiable Programs with Admissible Neural Heuristics, NeurIPS 2020.

### Environment

This code has been tested on the November 2021 x86_64 release of Anaconda3 (Python 3.9.7) and Pytorch 1.11.0. It should work without CUDA, but you will need to set the `device` variable in the training scripts to `'cpu'`. You will also need to install zss manually with `pip install zss`. This code will probably work on newer versions and/or architectures of Anaconda and Pytorch but it has not been tested on other configurations.

### Dataset

The dataset code + datasets themselves will be released at a later time due to size constraints. However, you should be able to adapt your dataset to the provided code with relative ease.

## Other

If you find our paper and/or code helpful, please consider citing it as

    @InProceedings{Tseng_2022_CVPR,
        author    = {Tseng, Albert and Sun, Jennifer J. and Yue, Yisong},
        title     = {Automatic Synthesis of Diverse Weak Supervision Sources for Behavior Analysis},
        booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
        month     = {June},
        year      = {2022},
        pages     = {2211-2220}
    }
