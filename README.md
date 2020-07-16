# L-1_BatchNorm
 Implementation of L1_BatchNorm by Hoffer et.al
 
This repository is an implementation of L1-BatchNorm proposed by Hoffer et.al in the paper "Norm matters: efficient and accurate normalization schemes in deep networks (https://arxiv.org/abs/1803.01814). They outline that the problem with L2-BatchNorm is that it invloves the operation of computing the variance across a batch of inputs and then inverting it. This operation is prone to numerical underflow or overflow when done in lower precision floating point numbers since it requires first computing the L2-Norm of (x-mean) and then inverting it after taking the square root. Hoffer et.al propose a L1-BatchNorm which simply computes a L1-Norm of (x-mean). This operation can be done in lower precision floating point numbers and thus can aid in speeding up the training and inference of a neural network.
 
L1-BatchNorm is tested against two variants of traditional batchnorm (one in which batchnorm layer is before the activation layer and one in which it is after). The dataset is Imagenette2-160 by fastai (https://github.com/fastai/imagenette) and code is based on fastaiv2 library.
 
The WandB report of the experiments can be seen here -> https://app.wandb.ai/a_bhimanyu/BatchNorm/reports/Testing-Various-Variants-of-BatchNorm--VmlldzoxNzEzNTc

