## Challenge: Can we beat the performance of the original model!

Trained on a subset of the ImageNet dataset (1.2 million training samples, 1000 output classes).
Random 227x227 (not 224x224 like the paper claims) patches of each image, and their corresponding horizontal reflections are extracted and used to augment the training dataset in order to prevent overfitting.

Image preprocessing is done in parallel on the cpu while training is on multiple GPUs.

Using ReLU makes the model train much faster than if they were using tanh. The post relu value is normalized by he sum of the squares of the hidden units in feature maps before and after the current feature map the hidden unit's in. Local response normalization is only applied to certain layers.

Overlapping pooling. s is the step size, z is the neighborhood size and they set s = 2, z = 3. The end of the current neighborhood and the start of next neighborhood share values, which adds a sense of continuity to the way pooling is done.

AlexNet splits the convolutional layers in half (top/bottom). Instead of a single 55x55x96 layer, they have 2 55x55x48 layers, one for each gpu.

Dropout roughly doubles the number of iterations needed to converge.

Model was trained using stochastic gradient descent with momentum.
Weights are initialized using a standard normal distribution with $\sigma = 0.01$.
Biases for layer 2, 3, 5 is initialized to 1, the biases for the rest of the layers is initialized to 0.

PCA:
- [Covariance](https://mathworld.wolfram.com/Covariance.html)
- [Eigenvalues and Eigenvectors](https://lpsa.swarthmore.edu/MtrxVibe/EigMat/MatrixEigen.html)
- **How to compute the characteristic equation of a matrix?**

Layers:
  - Input layer (227x227x3)

  - Convolutional, relu, dropout, output = 55x55x96
    - 96 11x11x3 kernels with a stride of 4 pixels
  - Response normalization
  - Max pooling, output = 27x27x96

  - Convolutional, relu, dropout, output = 27x27x256
    - 256 5x5x96 kernels with a stride of 1 pixel
  - Response normalization
  - Max pooling, output = 13x13x256

  - Convolutional, relu, output = 13x13x384
    - 384 3x3x256 kernels with a stride of 1 pixel

  - Convolutional, relu, output = 13x13x384
    - 384 3x3x384 kernels with a stride of 1 pixel

  - Convolutional, relu, output = 13x13x256
    - 256 3x3x384 kernels with a stride of 1 pixel

  - Max pooling, output = 6x6x256
    - Reshape to 9216x1

  - Linear, relu, input = 9216, output = 4096
  - Linear, relu, input = 4096, output = 4096
  - Linear, relu, input = 4096, output = 4096

  - Output layer, softmax, input = 4096, output = 1000
    - Cross entropy loss

Hyperparameters:
- Local response normalization: $k = 2, n = 5, \alpha = 10^{-4}, \Beta = 0.75$
- Batch size = 128
- Momentum = 0.9
- Weight decay = 0.0005 (reduces error)
- Learning rate = 0.01, manually reduced 3 times when the validation error rate stopped improving
- 90 cycles

At inference:
- Run the model on 10 variations of the image (top left, top right, bottom left, bottom right, center random 227x227 patches, plus all horizontally reflected versions) and average the output softmax values.
