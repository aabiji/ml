
- [x] Implement BatchNorm, Residual Connections and a smaller resnet model on a small dataset.
[CIFAR-10 dataset](https://cave.cs.toronto.edu/kriz/cifar.html)

- [ ] Implement U-Net on medical segmentation dataset and learn about the *Dice loss function*.
[DRIVE](https://github.com/openmedlab/Awesome-Medical-Dataset/blob/main/resources/DRIVE.md)

- [ ] Train U-Net on plant phenotyping datasets and answer the following questions.
[Crop/Weed Field Image Dataset](https://datasetninja.com/cwfid#download)
- How much data does U-Net need?
- Which normalization scheme works best (Batchnorm, GroupNprm, InstanceNorm, etc)?
- Which learning rate scheduler works best (StepLR, CosineAnnealingLR, ExponentialLR, etc)?
- How much of an effect does data augmentation actually have on model performance?
- Will transferring the weights of the same model architecture trained on a vegetation dataset (or just imagenet) improve accuracy and decrease the training time?
- ...

Resnet architecture in the paper:
Convolution on 3x32x32 input, 16x3x3 kernel

2 blocks (in = 16x32x32, out = 16x32x32), 16x3x3 kernel, padding = 1, stride = 1
2 blocks (in = 16x32x32, out = 32x16x16), 32x3x3 kernel, padding = 1, stride = 2, 1
2 blocks (in = 32x16x16, out =   64x8x8), 64x3x3 kernel, padding = 1, stride = 2, 1

Global average pooling (64x8x8 -> 64x1)
10 activation linear + softmax

Each block:
Batchnorm
ReLU
Convolution
Batchnorm
ReLU
Convolution
Residual
* Projection = 1x1 convolution with a stride of 2 (second and third block) applied to input
