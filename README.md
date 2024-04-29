# Involution
This repository contains re-Implementation of https://arxiv.org/abs/2103.06255

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

```models``` contains implementation of backbone models introduced the paper and related heads and neck architectures.

```notebooks``` contains main python notebooks for training and testing.

### Training

```
git clone https://github.com/thatblueboy/involution.git #clone the repo
git checkout submission_branch #change to the submission branch
```
Edit the ```train.py``` file. Here you can change various Hyperparameters in the config dict. Note that changing lr_scheduler will require corresponding chnage in lr_sceduler_kwargs.
Note: We use a random split split on Caltech256. For uniformity we store this split in the data_module.pth and load it for every training run.
This behaviour could be changed by setting the 'data_module_path' value in the config dict = None.

To switch from training to testing mode, change the last line in the train.py from
```
trainer.fit(model, data_module)
```
to
```
trainer.test(test, data_module)
```
after making all the necessary changes do:
```
python train.py
```
### To Do
