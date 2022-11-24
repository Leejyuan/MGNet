# MTCNet
Official implementation of MG-Net: Multi-level Global-Aware Network for Thymoma Segmentation
## Prerequisites
The neural network is developed with the PyTorch library, we refer to the PyTorch for the installation.

The following dependencies are needed:

Python = 3.6

pytorch = 1.8.1

numpy = 1.19.5

einops = 0.3.2

## Training

```python train_deep5.py -m UTNet -u train --reduce_size [2,4,8] --block_list 1234 --num_blocks 1,1,1,1 --gpu 0 --aux_loss```
