# Questions

- Why ReLU specifically?

# Neural network fundamentals

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

- The number of hidden units in a shallow network is called the *network capacity (p.29 figure 3.6)*. Sign matters when scaling neural network parameters!
  With enough capacity (hidden units), the network can approximate any function at a given resolution (Universal approximation theorem). This is because
  as you add more hidden units, they describe smaller and smaller portions of the function that is more accurately described by a line. (*p. 30, figure 3.5*)

- To have a multivariate output, express the parameters that weigh the hidden units as a matrix instead of a vector
- To have a multivariate input, add more terms to the hidden unit's linear function

- To visualize multivariate inputs, the dimension of the output is the number of inputs plus 1 (the input is an axis and each output is an axis) (*p.32 figure 3.8*).
  Which is why non trivial neural networks are impossible to visualize. (*p.34 figure 3.10*)
  For example, with a network with 1 output and 1 input, the x and y axes could correspond to the first and second input, and the depth (z) could correspond to the function output,
  making a convex 3d surface with the crevasses defined by the places where the hidden units are clipped by the activation function.

- The hidden units are defined as their local offset plus the sum of all the weighed inputs. The network output is then defined as the global
  offset for a particular output plus the sum of all the weighed hidden units. (*p.33 equation 3.11 and p.35 equation 3.12*). Input weights and
  hidden unit weights are matrices.

- Basic terminology: (*p.36, figure 3.12*)

- A polytope is an N-dimensional object with flat faces. (ex: a polygon is a 2d polytope, a polyhedron is a 3d polytope).

- Binomial coefficient is a positive integer describing how many times $k$ items can be chosen from `n` items.

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

- *Parameters* are input/hidden layer weights and biases, while *hyperparameters* are hidden unit values.

- The equations that govern a deep neural network are easily vectorized (p.48, *figure 4.6*).
  Let $y$ be the output, $h$ be a network layer, $K$ be the number of layers, exlcuding the input and output layres, $D_j$ be the number of hidden units at layer $j \in [1, K]$, $D_i$ be the number of network inputs, and $D_o$ be the number of network outputs.  The bias matrix has dimension $D_{j + 1} \times 1$, and the weight matrix has dimension $D_{j + 1} \times D_{j}$.
$$\bold{h_j} = a[\boldsymbol{\beta_{j - 1}} + \boldsymbol{\Omega_{j - 1}}\boldsymbol{h_{j - 1}}]$$
$$\bold{y} = \boldsymbol{\beta_j} + \boldsymbol{\Omega_j}\boldsymbol{h_j}$$

- Deep neural networks can approximate any given function to any specified accuracy, just like shallow neural networks, and
  because of the added depth (number of hidden layers), deep neural networks can approximate functions much more efficiently than shallow neural networks.

- We can create a lot of linear regions with few parameters, but they depend on each other in ways that quickly become hard to understand.

# Probability

- $Pr(x)$, a [probability density function](https://medium.com/@kavya8a/understanding-probability-density-functions-a-beginners-guide-06c9821ed5c0) (PDF), assigns a non negative probability *density* to every value in the input domain. $\int Pr(x) \,dx = 1$. We are assuming that random variables are continuous.

- A joint probability distribution (ex: $Pr(x, y)$, probability of $x$ and $y$) assigns a probability to every *combination* of values. $\int \int Pr(x, y) \,dxdy = 1$, the sum of the probability distributions of all the variables needs to equal 1.

- A marginal distribution is the sum of the probabilities of a particular value for every other value, for every value. For example with $Pr(x, y)$, the marginal distribution $Pr(x)$, is a probability distribution where every probability associated to a particular $x$ is the sum of the probability of $x$ occuring at at each y value. For example, if we have a joint probability distribution of height (h) and weight (w), $Pr(h, w)$, then $Pr(h)$ is the probability distribution of height at any given weight. Marginalization allows us to move from joint probability space to an independent variable space.

- A conditional probability $Pr(x|y)$ is the probability of x given y. For example, if you wanted the probability of someone being 6ft tall given that they are 200 lbs, you would divide the probability that someone is 6ft tall and 200 lbs and divide that by that the probability that someone is 200 lbs.
$$Pr(x|y) = \frac{Pr(x, y)}{Pr(y)}$$
