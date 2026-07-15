
- [x] Implement BatchNorm, Residual Connections and a smaller resnet model on a small dataset.
[CIFAR-10 dataset](https://cave.cs.toronto.edu/kriz/cifar.html)

- [ ] Implement U-Net on medical segmentation dataset and learn about the *Dice loss function*.
[DRIVE](https://github.com/openmedlab/Awesome-Medical-Dataset/blob/main/resources/DRIVE.md)

- [ ] Train U-Net on plant phenotyping datasets and answer the following questions.
[Crop/Weed Field Image Dataset](https://datasetninja.com/cwfid#download)
- U-Net uses valid convolutions as to not introduce any extra information. However, this makes the model's output size
  much smaller than its input size. What if we used padded convolutions to preserve the data dimensions? How much would
  that degrade model performance? Perhaps it's possible to add a correcting term to the loss function in order to
  compensate for the added information and maintain similarly high levels of accuracy?

- How much noise or variability does the decoder introduce into the output segmentation map? Information is lost when
  layers from the encoder are cropped and concatenated to layers in the decoder. So, how much does the output segmentation
  map ressemble the input image? Can there ever be a situation where the segmentation map looks nothing like the input image?

- How many training samples does U-Net really need, with or without data augmentation?

- How much of an effect does data augmentation actually have on model performance?
- Which normalization scheme works best (Batchnorm, GroupNprm, InstanceNorm, etc)?
- Which learning rate scheduler works best (StepLR, CosineAnnealingLR, ExponentialLR, etc)?
- ...

---

U-Net architecture:

A U-Net is split into an encoder and a decoder, which are mirrored. Layers in the encoder halve spatial dimensions while doubling feature sizes.
With larger receptive fields, increasingly abstract semantic representations of the data can be learned at the cost of sacrificing localization
precision. Layers in the decoder double spatial dimensions while halving feature sizes. This allows the network to localize the semantic
reprsentation at increasingly larger spatial dimensions. Corresponding layers in the encoder are cropped and concatenated to the upsampled layers
in the decoder in order to reintroduce most of the fine spatial information that was preserved in the encoder. Subsequent convolutions learn how
the fine spatial information and the coarse semantic information can be combined effectively. The concatenation also gives layer inputs multiple
shorter paths to reach the output, helping information flow through the network. All of these design decisions result in the encoder learning
what is in the data and the decoder learning where it is spatially.

Each label corresponds to its own channel, so the final layer in the model uses a 1x1 convolution to map the 64 output channels to the target
number of channels. Softmax is applied per pixel, summing across channels. Cross entropy loss is used as the loss function. Each pixel is
chosen from the channel that coressponds to the pixel's true label.

\[ E = \sum_{x \in \Omega} w(\mathbf{x}) log(p_{l(x)}(\mathbf(x))) \]

https://en.wikipedia.org/wiki/Normal_distribution

??
- Overlap-tile strategy -> could this be related to the fact that U-Net's output is smaller than its input?
- Semantic deformation

