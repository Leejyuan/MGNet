# MGNet
Official implementation of MG-Net: Multi-level Global-Aware Network for Thymoma Segmentation
## Prerequisites
The neural network is developed with the PyTorch library, we refer to the PyTorch for the installation.

The following dependencies are needed:

Python = 3.6

pytorch = 1.8.1

numpy = 1.19.5

einops = 0.3.2

## Training

```python train.py -m MGNet -u train --reduce_size [2,4,8] --block_list 1234  --gpu 0 --aux_loss --fuse```

To optimize MGNet in your own task, there are several hyperparameters to tune:

'--block_list': indicates apply transformer blocks in which resolution. The number means the number of downsamplings, e.g. 3,4 means apply transformer blocks in features after 3 and 4 times downsampling. Apply transformer blocks in higher resolution feature maps will introduce much more computation.

'--reduce_size': indicates the size of downsampling for efficient attention. reduce_size can be adjusted according to the resolution of feature maps and the size of foreground.

'--fuse': whether fuse feature maps in decoder blocks for prediction.

'--aux_loss': whether applie deep supervision in training

## Testing

```python test.py -m MGNet -u test --reduce_size [2,4,8] --block_list 1234  --gpu 0 --aux_loss --fuse --pred --resume```

'--pred': the path to save predict results

'--resume': the path of checkpoint
