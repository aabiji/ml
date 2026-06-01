## Questions
- How is SGD an implicit regularizer for smooth functions?
- Are there efficient hyperparameter optimization algorithms?

## Interesting
- Distributed model training
- Reducing the memory requirements of a model during training
- Double descent

## Math

- $Pr(x)$, a [probability density function](https://medium.com/@kavya8a/understanding-probability-density-functions-a-beginners-guide-06c9821ed5c0) (PDF), assigns a non negative probability *density* to every value in the input domain. $\int Pr(x) \,dx = 1$. We are assuming that random variables are continuous.

- A joint probability distribution (ex: $Pr(x, y)$, probability of $x$ and $y$) assigns a probability to every *combination* of values. $\int \int Pr(x, y) \,dxdy = 1$, the sum of the probability distributions of all the variables needs to equal 1.

- A marginal distribution is the sum of the probabilities of a particular value for every other value, for every value. For example with $Pr(x, y)$, the marginal distribution $Pr(x)$, is a probability distribution where every probability associated to a particular $x$ is the sum of the probability of $x$ occuring at at each y value. For example, if we have a joint probability distribution of height (h) and weight (w), $Pr(h, w)$, then $Pr(h)$ is the probability distribution of height at any given weight. Marginalization allows us to move from joint probability space to an independent variable space.

- A conditional probability $Pr(x|y)$ is the probability of x given y. For example, if you wanted the probability of someone being 6ft tall given that they are 200 lbs, you would divide the probability that someone is 6ft tall and 200 lbs and divide that by that the probability that someone is 200 lbs.
$$Pr(x|y) = \frac{Pr(x, y)}{Pr(y)}$$

- *Variance* is the average squared distance from the mean.

- *[KL Divergence](https://www.datacamp.com/tutorial/kl-divergence)* is a measure of how much the test distribution diverges
  from the correct distribution. Remember that entropy is just a weighted sum of surprisal. We subtract cross entropy from entropy.
  Ideally, the two distributions $p(x)$ (actual) and $q(x)$ (test) are equal, so the entropy and the cross entropy are equal, and the divergence is zero.
  $$D_{KL}[P || Q] = \int_{-\infty}^{\infty} p(x)\log{\frac{P(x)}{Q(x)}}$$

- The *[Dirac Delta Function](https://math.libretexts.org/Bookshelves/Differential_Equations/Introduction_to_Partial_Differential_Equations_(Herman)/09%3A_Transform_Techniques_in_Physics/9.04%3A_The_Dirac_Delta_Function)* is +infinity at 0 and 0 everywhere else. The integral of the function over its entire domain is 1. It's useful for modelling impulses.

- A *Jacobian* matrix is a matrix storing the derivative of one vector with respect to another. For example:
$$ f(x, y) = \begin{bmatrix} x + y \\ xy \end{bmatrix} $$
$$
J = \frac{\partial f}{\partial (x, y)}
  = \begin{bmatrix}
    \frac{\partial f_1}{\partial x} & \frac{\partial f_1}{\partial y} \\
    \frac{\partial f_2}{\partial x} & \frac{\partial f_2}{\partial y} \\
    \end{bmatrix}
  = \begin{bmatrix} 1 & 1 \\ y & x \end{bmatrix}
$$

- The *sigmoid* function maps a real value $z$ to a range of $[0, 1]$.
$$\text{sig}[z] = \frac{1}{1 + \text{exp}[-z]}$$

- The *softmax* function maps a vector of real values (logits) $\bold{z}$ to a vector of probabilities in a range of 0 to 1, where all probabilities sum to 1.
$$\text{softmax}[\bold{z_k}] = \frac{\exp[z_k]}{\sum_{k' = 0}^{K} \exp[z_k']}$$

## Overview

- A model is a mathematical function defined by parameters that map an input to an output.
  Each parameter depends on a corresponding input value.
  To "train" the model you define a cost function (*p.19 figure 2.5*) that measures how close the model's output is to the expected output.
  A loss function commonly used is *least squared loss*.
  You then find the parameters that will minimize the cost function (*p.21 figure 2.6*), and replace the old model parameters with the new ones.
  Over time, the loss should get smaller and smaller.
  There's a risk that the model underfits by not capturing a relationship between inputs and outputs.
  There's a risk tha the model overfits by performing well on the training samples, but falling apart on real world samples.

- There are *discriminative* models and *generative* models. Discriminative models find outputs from inputs,
  while generative models find inputs from outputs.
  Most models are discriminative since it's easier to exploit the prior knowledge that's baked into the input data.
  Generative models have much higher losses than discriminative models, and seem to perform worse on the demo examples.
  Trying to capture all of the nuance that goes into generating a specific output really is so much harder than mapping an input to an output.
  It might be the same local minimum yes, but it's so much harder to get to.

- The most naive way to train the model is to evaluate the model on each possible permutation of parameters and pick the paramters that
  produce the lowest loss. That's obviously a stupid idea though.

- Without an activation function, the model can only represent linear functions, which is very limiting. Even if you reformulate neural networks
  to use a polynomial of a higher degree, or some transcendal function, the model's complexity is still fundamentally capped. Using an activation
  function breaks that constraint and allows the model to represent non linear functions.
  An activation function has to be non linear in order to be effective.
  A common activation function is ReLU (Rectified Linear Unit). If the value is <= 0, the output is 0, else it's the value, which clamps negative numbers.

- The number of hidden units in a shallow network is called the *network capacity (p.29 figure 3.6)*. The theoretical capacity is called the
  *representational capacity*, while the actual number of functions that the model can approximate is called the *effective capacity*.
  With enough capacity (hidden units), the network can approximate any function at a given resolution (Universal approximation theorem). This is because
  as you add more hidden units, they describe smaller and smaller portions of the function that is more accurately described by a line. (*p. 30, figure 3.5*)

- To visualize multivariate inputs, the dimension of the output is the number of inputs plus 1 (the input is an axis and each output is an axis) (*p.32 figure 3.8*).
  Which is why non trivial neural networks are impossible to visualize. (*p.34 figure 3.10*)
  For example, with a network with 1 output and 1 input, the x and y axes could correspond to the first and second input, and the depth (z) could correspond to the function output,
  making a convex 3d surface with the crevasses defined by the places where the hidden units are clipped by the activation function.

- The hidden units are defined as their local offset plus the sum of all the weighed inputs. The network output is then defined as the global
  offset for a particular output plus the sum of all the weighed hidden units. (*p.33 equation 3.11 and p.35 equation 3.12*). Input weights and
  hidden unit weights are matrices.

- Basic terminology: (*p.36, figure 3.12*)

- A polytope is an N-dimensional object with flat faces. (ex: a polygon is a 2d polytope, a polyhedron is a 3d polytope).

- Binomial coefficient is a positive integer describing how many times $k$ items can be chosen from $n$ items.

- By adding more layers to the network, we're *composing* functions.
  For example, with a single hidden layer, the model can be thought of as $y = f(x)$.
  With a second hidden layer, the model can be thought of as $y = g(f(x))$, where $x$ is the input to the first hidden layer,
  f(x) is the input to the second hidden layer and g(x) is the model output.
  Some values of x will give the same value of f(x). Some values of f(x) will give the same value of g(x). So, large sets of $x$
  will give the same network output.

- By redefining the weights/biases for the previous layer's hidden units, we are no longer bound by them. Which means that
  the hidden units in the previous layer can be used as inputs in the current layer (p.43, *equations 4.5, 4.6*).

- Can think of deep neural networks in 2 main ways:
  - Hidden layers fold the input space (multiple inputs map to one output), which shows how the final output depends on the initial input and the layer outputs
  - Hidden layers clip their inputs, which shows how joints are created in the output space

- *Parameters* are input/hidden layer weights and biases, while *hyperparameters* are high level settings on how the neural
  network operates (ex: learning rate, epoch, batch size, number of layers, hidden units per layer, etc).
    - A common way to optimize hyperparameters is to split the training dataset into three subsets. A training set,
      a validation set, which is used to choose the hyperparameters that lead to the best model performance, and a
      test set which is used for a final model performance evaluation.

    - Another common apporoach is called k-fold cross validation. Training samples are split into $N$ random subsets,
      and for $N$ iterations, training is done with $N - 1$ subsets while hyperparameter validation is done with
      the remaining subset. The final model performance is evaluated using an average of the $N$ model predictions
      on a different test dataset.

    - There are many other hyperparameter algorithms.

- The equations that govern a deep neural network are easily vectorized (p.48, *figure 4.6*).
  Let $y$ be the output, $h$ be a network layer, $K$ be the number of layers, exlcuding the input and output layres, $D_j$ be the number of hidden units at layer $j \in [1, K]$, $D_i$ be the number of network inputs, and $D_o$ be the number of network outputs.  The bias matrix has dimension $D_{j + 1} \times 1$, and the weight matrix has dimension $D_{j + 1} \times D_{j}$.
$$y = f[\bold{x}, \phi]$$
$$\phi \in [\boldsymbol{\beta}, \boldsymbol{\Omega}]$$
$$\bold{h_j} = a[\boldsymbol{\beta_{j - 1}} + \boldsymbol{\Omega_{j - 1}}\boldsymbol{h_{j - 1}}]$$
$$\bold{y} = \boldsymbol{\beta_j} + \boldsymbol{\Omega_j}\boldsymbol{h_j}$$

- Deep neural networks can approximate any given function to any specified accuracy, just like shallow neural networks, and
  because of the added depth (number of hidden layers), deep neural networks can approximate functions much more efficiently than shallow neural networks.

- We can create a lot of linear regions with few parameters, but they depend on each other in ways that quickly become hard to understand.

- Gaussian distribution: univariate normal distribution.

- A model no longer computes an output directly, but instead computes a set of parameters ($\theta$) that describe *a probability distribution*.
  Minimizing the loss function $L[\phi]$ means maximizing $Pr(y|\theta)$.
  For example, if we know that our output ($y$) follows a Gaussian distribution, then we'll make the model output the distribution's mean. The goal would be
  that the model outputs $y$ as the mean value, maximizing $Pr(y|\theta)$. To do this, we would want to minimize the distance between $y$ and the
  mean, ... mean squared error!

- In general, we want to choose network parameters that satisfy the maximum likelyhood criterion of our distribution. Parameters that will
  maximize the probability that the model produces the expected output given an input across all training samples. The following are three different
  ways of saying the same thing (where $I$ is the number of samples).
$$
\begin{align}
\boldsymbol{\hat{\phi}} &= \underset{\phi}{\text{argmax}}[\prod_{i = 1}^{I} Pr(y_i|x_i)]\\
                        &= \underset{\phi}{\text{argmax}}[\prod_{i = 1}^{I} Pr(y_i|\theta_i)]\\
                        &= \underset{\phi}{\text{argmax}}[\prod_{i = 1}^{I} Pr(y_i|f[\bold{x_i}, \boldsymbol{\phi}])]
\end{align}
$$
- It's more practical to take the sum of the log of all the probabilities, since they may be very small. Also, by convention we want our loss function
  to minimize loss, which can be acheived by multiplying the sum by -1 and using argmin. This is the *negative log-likelihood*!
$$\boldsymbol{\hat{\phi}} = \underset{\phi}{\text{argmin}}[-\sum_{i = 1}^{I} \log[Pr(y_i|f[\bold{x_i}, \boldsymbol{\phi}])]]$$
- During inference, our output value is the maximum of the output probability distribution.

- A *heteroscedastic* model is one where its uncertainty varies with its input. A homoscedastic model is one where its uncertainty is constant.

- There a three main sources of error when training a model: The extent to which model performance is maintained on the training set is called *generalization*.
  - *Noise*: The inherent randomness (mislabelling, multiple inputs mapping to the same output, etc) in the training dataset is insurmountable.

  - *Bias*: The model might not have enough capacity to fit the data accurately.
            Increasing the model capacity decreases bias but increases variance, and doesn't necessarily decrease test error.
            This is known as the *bias-variance tradeoff*. Increasing model capacity unfortunately makes it model noise in the training data
            better, which is referred to as *overfitting*.

  - *Variance*: The fitted function will vary slightly based off of the training samples (and possibly SGD). Increasing the number of training samples decreases variance.

- As model capacity increases, test error should decrease up until a point where variance
  becomes too large, the model starts to overfit and the test error starts to increase. However, for many datasets,
  the test error starts decreasing again (*p.130 figure 8.10*) as the model capacity continues to increase. This is
  referred to as [*double descent*](https://www.lesswrong.com/posts/FRv7ryoqtvSuqBxuT/understanding-deep-double-descent).
  At the moment we don't fully know why, but there's at least one plausible explanation. As the model capacity increases
  and the training error approaches zero, the model's fitting function is able to pass through every single training sample.
  The segments between data points are initially very erratic, leading to a large test error. However, as the model capacity
  continues to grow, the segments between data points become smoother and smoother, which makes the model fit to new test
  points better, decreasing test error. This is especially plausible when you consider that in a high dimensional input space,
  the training samples are extremely sparse (curse of high dimensionality). There's a large number of possible fitting
  functions passing through all training samples that the model can adopt. The model's tendency to adopt certain families
  of fitting functions is called *inductive bias*. Any factor that biases a solution to a set of equivalent solutions is
  called a *regularizer*. The hypothesis is that SGD is an implicit regularizer of smooth functions.

- Example of *one-hot encoding*: `[0.1, 0.2, 0.3, 0.9, 0.7] -> [0, 0, 0, 1, 0]`

**Model capacity of 4???**

## Loss functions
- *Mean squared error* is used when you assume that the output follows a Gaussian distribution (ex: regression problems).
  When performing inference, the model output is $\hat{\bold{y}} = f[\bold{x}, \boldsymbol{\phi}]$, since the model produces
  the mean of the output Gaussian distribution, which just happens to be the distribution's maximum. In addition to using the mean
  in the loss function, we could also add variance, which is this case measures how certain the output is (high variance = high
  uncertainty since the distribution would be spread out more. Normalization is done to make the loss independent of the number
  of training samples, and to make comparing loss across datasets easier. $I$ is the number of training samples.
  $$L[\phi] = \frac{1}{I} \sum_{i = 1}^{I} (\bold{y_i} - f[\bold{x_i}, \boldsymbol{\phi}])^2$$

- *Binary cross-entropy loss* is used in binary classification problems. A suitable probability distribution would be a
  Bernouilli distribution, where the probability of the first option is $\lambda \in [0, 1]$, and the probability of the
  second option is $1 - \lambda$. The model outputs $\lambda$. Sigmoid is used since it can't be guaranteed that said
  output will lie between 0 and 1. During inference, we assume that if $\lambda > 0.5$, then the output is 1, else it's 0.
  $$L[\phi] = \sum_{i = 1}^{I} -(1 - y_i)\log[1 - \text{sig}[f[\bold{x_i}, \boldsymbol{\phi}]]] - y_i\log[\text{sig}[f[\bold{x_i}, \boldsymbol{\phi}]]]$$

- *Multiclass cross-entropy loss* is used in multiclass classification problems. A suitable probability distribution would be a categorical
  distribution, where the probability of each class is $\lambda_k \in [0, 1], k \in [1, K]$, where $K$ is the number of categories. All
  probabilities must sum up to 1. Softmax is applied to the function's output to get the categorical distribution.
  $$L[\phi] = -\sum_{i = 1}^{I} (f_{yi}[\bold{x_i}, \boldsymbol{\phi}] - \log[\sum_{k = 1}^{K} \exp[f_k[\bold{x_i}, \boldsymbol{\phi}]]])$$
  Sum the difference between the expected output and the sum of $exp[z_k]$ for every class in the model's output, for each training sample.

- When we have a multivariate model output, we treat each output and each error as *independent*.
  Each output gets its own loss function, which can vary. Thus, the model's loss function becomes a sum of all the loss functions for each output.

- See the chart on *p.70* for more.

## Common optimization algorithms
- The goal of an optimization algorithm is to update the model's parameters as to minimize the loss. This process is known as learning, training or fitting.

- *Gradient descent*: Compute gradients. Gradients are the partial derivatives of the loss function with respect to the model parameters.
  Then step backwards. $\alpha$ is the *learning rate*, which is a hyperparameter that controls how quickly the parameters change.
  At the start of training, the learning rate should be relatively large, (the model should take large steps around the parameter space),
  and as training draws to a close it should decrease (the model should take small steps around the minimum). *Line search* can be performed
  to determine which learning rate will make the model learn the fastest.
  The problem with gradient descent is that we can't predict whether the model will converge on a local minimum or a
  global minimum, or get stuck in a *saddle point* (flat area of the function) especially when the model function is non-convex.
  $$\boldsymbol{\hat{\phi}} = \boldsymbol{\phi} - \alpha\frac{\partial L}{\partial \boldsymbol{\phi}}$$

- *Stochastic gradeitn descent*: SGD adds randomness to the stepping process by sampling a random subset of training samples, a *batch*.
  $N / B$ batches are sampled every *epoch*, where $N$ is the number of training samples and $B$ is the number of training samples in a batch.
  Because of the batching, SGD is more efficient, and since the loss function for each batch is different, and thus the gradients are different,
  the model takes a noiser path to convergence. This allows the parameters to potentially step out of a local minima or a saddle point.
  $$\hat{\boldsymbol{\phi}} = \boldsymbol{\phi} - \alpha \sum_{i \in B_t} \frac{\partial l_i[\boldsymbol{\phi_i}]}{\partial \boldsymbol{\phi}}$$
  Where $B_t$ is the current batch of training samples and $l_i$ is the loss given the chosen training samples.

- *Nesterov accelerated momentum*: By adding a weighted combination of the previous gradients, and the gradients at the target position, the model can converge on a minimum much faster. Notice that the momentum terms are recursive, and older momenta get weighed less and less. Past gradients are weighed more than the current gradients. By computing the gradient are the target position we also avoid the overshooting that would happen if we used the gradients are the starting position.
$$m_{t + 1} = \Beta * m_t + (1 - \Beta)\sum_{i \in B_t} \frac{\partial l_i[\boldsymbol{\phi_t} - \alpha\Beta * m_t]}{\partial \boldsymbol{\phi}}$$
$$\boldsymbol{\hat{\phi}} = \boldsymbol{\phi_t} - \alpha * m_{t + 1}$$

- *Adam*: The Adam optimizer takes ideas from several other algorithms and combines them into one. First, Include a weighted history of both the momentum and the squared momentum. Second, amplify the momentum and squared momentum when $t$ is small, doing nothing when $t$ is large. This is because the momentum (which is initialized to 0) will be too small when t is small. Lastly, normalize the gradients so that the parameters step the same distance in each direction.
$$
m_{t + 1} = \Beta * m_t + (1 - \Beta)\sum_{i \in B_t} \frac{\partial l_i[\boldsymbol{\phi_t}]}{\partial \boldsymbol{\phi}}\\
v_{t + 1} = \gamma * v_t + (1 - \gamma) \left(\sum_{i \in B_t} \frac{\partial l_i[\boldsymbol{\phi_t}]}{\partial \boldsymbol{\phi}}\right)^2\tag{1}
$$
$$
\tilde{m_{t + 1}} = \frac{m_{t + 1}}{1 - \Beta^{t + 1}}\\
\tilde{v_{t + 1}} = \frac{v_{t + 1}}{1 - \gamma^{t + 1}}\tag{2}
$$
$$\boldsymbol{\hat{\phi}} = \boldsymbol{\phi_t} - \alpha * \frac{\tilde{m_{t + 1}}}{\sqrt{\tilde{v_{t + 1}}} + \epsilon}\tag{3}$$

## Backpropagation
- Backpropagation is an algorithm to iteratively apply the chain rule to any arbitrary computational graph. A neural network can be thought of as a series of function compositions, where each layer composes the previous layer. A neural network can also be thought of as a [computation graph](https://www.youtube.com/watch?v=i94OvYb6noo) (DAG), where each layer is a node. Backpropagation is just going through each node, right to left, and computing the local gradient, multiplying it by the global gradient (which is initialized to 1), and passing an updated global gradient to child nodes (previous layers). Gradients are computed for every training sample in a batch, then summed together to get the gradient used in an optimization algorithm.

- A *forward pass* is done to compute and cache values of hidden units in each layer, which will be used to compute gradients:
$$
\begin{align*}
f_0       &= \Beta_0 + \Omega_0\bold{x}\\\
h_k       &= a[f_{k - 1}] && k \in \{1, 2, 3, ... K\}\\
f_{k - 1} &= \Beta_k + \Omega_k \bold{h_k} && k \in \{1, 2, 3, ... K\}
\end{align*}
$$

- A *backward pass* is done to compute the derivative of the loss with respect to every weight and bias in the network. The rules for computing local gradients and updating the global gradient are as follows (*p. 106, equation 7.25*):
$$\frac{\partial l_i}{\partial \Beta_k} = \frac{\partial l_i}{\partial f_k}$$
$$\frac{\partial l_i}{\partial \Omega_k} = \frac{\partial l_i}{\partial f_k} h_k^T$$
$$\frac{\partial l_i}{\partial f_{k - 1}} = I[f_{k - 1} > 0] \odot \Omega_k^T \frac{\partial l_i}{\partial f_k}$$
$$\frac{\partial l_i}{\partial \Beta_0} = \frac{\partial l_i}{\partial f_0}$$
$$\frac{\partial l_i}{\partial \Omega_0} = \frac{\partial l_i}{\partial f_0} \bold{x_i}^T$$

- $k \in \{K - 1, K - 2, K - 3, ... 1\}$, and $\frac{\partial l_i}{\partial f_k}$ is the the global gradient. The derivative of ReLU with respect to the hidden units is a Jacobian matrix ($J$). When $i \neq j, J_{ij} = 0$, since $ReLU^{\prime}(i)$ only depends on $i$. The diagonal elements in $J$ would be either 1 or 0 (considering the definition of ReLU). Instead of computing the entire Jacobian, its diagonal elements are directly computed, which is just: `activation_derivative_vector = hidden_unit > 0`.

- [Expectation](https://www.probabilitycourse.com/chapter3/3_2_2_expectation.php) is a weighted sum of all the possible values a random value could be. If the expectation of the magnitude of the weights are smaller than 1, the gradient vanishes, and if the expectation of the magnitude of the weights are larger than 1, the gradient explodes. Both problems can occur in the forward or backward pass. A solution to this is *He initialization* (also known as *Kaiming initializtion*). Initializing the weights to random values taken fro a normal distribution centered around 0, with a variance $(1)$, where $D_h$ is the dimension of the current layer, and $D_h\prime$ is the dimension of the next layer, keeps the gradients stable. This particular initialization is used with the ReLU activation function.
If the current and next layer have the same dimension, then an average of the variance for the forward pass and backward pass doesn't have to be used $(2)$.
$$\sigma^2 = \frac{4}{D_h + D_{h\prime}}\tag{1}$$
$$\sigma^2 = \frac{2}{D_h}\tag{2}$$
