# Involution
This repository contains re-Implementation of https://arxiv.org/abs/2103.06255

### Involution Kernal

<img src="https://github.com/thatblueboy/involution/assets/100462736/43bca57d-6cee-403d-91d8-68cc83a97a86" width="45%"></img> <img src="https://github.com/thatblueboy/involution/assets/100462736/cc45de86-83c1-4eb7-abb8-e48f4b8f0637" width="45%"></img> 


The authors of _Involution: Inverting the Inherence of Convolution for Visual Recognition_ propose a novel involutional layers, which aims to enhance the representation power of convolutional neural networks by inverting the inherent properties of convolution operations. As such these kernals are channel agnostic and spatial specific.

### Usage

```python
from models.rednet import RedNet

rednet50 = RedNet(50) # 50 layer Rednet model
```
### Folders

```models``` contains implementation of backbone models introduced the paper and related heads and neck architectures.

```notebooks``` contains main python notebooks for training and testing.

### To Do
