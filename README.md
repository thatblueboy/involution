# Involution
This repository contains re-Implementation of the paper https://ieeexplore.ieee.org/document/9577788

### Involution Kernal

<img src="https://github.com/thatblueboy/involution/assets/100462736/43bca57d-6cee-403d-91d8-68cc83a97a86" width="45%"></img> <img src="https://github.com/thatblueboy/involution/assets/100462736/cc45de86-83c1-4eb7-abb8-e48f4b8f0637" width="45%"></img> 


The authors of _Involution: Inverting the Inherence of Convolution for Visual Recognition_ propose a novel involutional layers, which aims to enhance the representation power of convolutional neural networks by inverting the inherent properties of convolution operations. As such these kernals are channel agnostic and spatial specific.

### Setup
```
pip install torch torchvision
pip install wandb
pip install lightning
```

### Folders

```models``` folder contains the main backbone implementations of models used as well as classification heads and lightning class for easy training and logging

```slides``` contains presentation slides with results on Caltech-256

```data``` contains the data module and custom dataset

### Training

```
git clone https://github.com/thatblueboy/involution.git #clone the repo
```

Following model and training parameters can be configured in ``` train.py ``` by modifying the configs dictionary

#### Parameters

- ``` model ``` to specify which model you want to train. **ResNetClassifier** for Resnets and **RedNetClassifier** for RedNets containing involutions.

- ```ReDSnet_type``` to specify depth of the model. Can be one of **26**, **38**, **50**, **101**, **152**

- ```batch_size``` is training batch size 

- ```optimizer``` and ```optimizer_kwargs``` for learing optimizer. optimizer can be **Adam** or **SGD**

- ```num_workers``` is number of workers

- ```lr_scheduler``` for learing rate scheduler. One of **ExponentialLR**, **CosineAnnealingLR**, **LinearLR**, **StepLR**, **PolynomialLR**. Any changes to the lr_scheduler will require corresponding changes to ```lr_sceduler_kwargs```  

#### Note
 We use a random split split on Caltech256. For uniformity we store this split in the data_module.pth and load it for every training run. This behaviour can be changed by setting the ```'data_module_path'``` value in the config dict to ```None```.

#### Switch from train to test
- To switch from training to testing mode, change the last line in the train.py from
```
trainer.fit(model, data_module)
```
to
```
trainer.test(model, data_module)
```

#### After making all the necessary changes:
```
wandb login
python train.py
```

### Acknowledgements

Code was heavily inspired by the original papers code: https://github.com/d-li14/involution

Original paper can be found here: https://ieeexplore.ieee.org/document/9577788

This project was done as a partial fulfillment of the course CS F425: Deep Learning at BITS-Pilani
