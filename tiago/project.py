#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Thu 21 Mar 2013 15:03:35 CET

"""Exercise 3: complete the missing bits in the functions bellow, so the
program ``ex3.py`` works flawlessly. See more details and test cases at our
course slides.

N.B.: You should try to use the vectorized form in this exercise, or your
program will be too slow. M-NIST has 60k input vectors for training, each with
a dimensions 784 (28x28 images).
"""

import numpy
import scipy.optimize

class Machine:
  """Represents a Multi-Layer Perceptron Machine"""

  def __init__(self, nhidden=2, regularization=0):
    """Initializes the MLP with a number of inputs and outputs. Weights are
    initialized randomly with the specified seed.

    Keyword parameters:

    seed
      The random seed to use when initializing the weights

    nhidden
      The number of units on the hidden layer

    regularization
      A regularization parameter to be passed to the MLP.
    """

    # Create two matrices of random weights, representing w1 and w2
    # The '+1' comes from the extra entry for the bias unit.
    # In doubt, run the maths yourself.
    # z1 = w1^T * X
    # z2 = w2^T * a1
    self.w1 = numpy.random.rand((28*28)+1, nhidden)
    self.w2 = numpy.random.rand(nhidden+1, 10)

    self.regularization = regularization

  def save(self, h5file):
    h5file.set('w1', self.w1)
    h5file.set('w2', self.w2)

  def load(self, h5file):
    self.w1 = h5file.read('w1')
    self.w2 = h5file.read('w2')

  def activation(self, z):
    """Computes the activated output of the neuron."""

    # If you want to use a sigmoid (logistic) function, uncomment the line
    # below. Also remember to change ``activation_prime()`` below.
    #
    # WARNING: In this case, you also need to feed in, values between the range
    # [0, 1] to this network. Make sure you modify your calling program to
    # account for this!

    #return 1 / (1 + numpy.exp(-z))

    return numpy.tanh(z)

  def activation_prime(self, a):
    """Returns the derivative of the tanh w.r.t. a given layer"""

    # If you want to use a sigmoid (logistic) function as activation, uncomment
    # the code below for the matching derivative. Also remember to change the
    # method ``activation()`` above.
    #
    # WARNING: In this case, you also need to feed in, values between the range
    # [0, 1] to this network. Make sure you modify your calling program to
    # account for this!

    # logistic'(z) = logistic(z) * (1 - logistic(z)) = a * (1-a)
    #return a * (1 - a)

    # tanh'(a) = 1 - tanh(z)**2 = 1 - a**2
    return 1 - a**2

  def J(self, X, y):
    """Calculates the cost J, with regularization

    This version uses the Mean Square Error (MSE) as the cost with
    regularization:

      J_MSE(weights) = 0.5 * mean( (MLP(X)-y)^2 ) + ...
                          ... + (0.5 * lambda / N) * (sum(w1**2) + sum(w2**2))

    Keyword attributes:

    X
      The input vector containing examples organized in columns. The input
      matrix does **not** contain the bias term.

    y
      The expected output for the last layer of the network. This is a simple 2D
      numpy.ndarray containing 1 column vector for each input example in the
      original input vector X. Each column vector represents one output.

    Returns the ML (logistic regression) cost with regularization.
    """

    # If you would like to use the "Logistic Maximum Likelihood"-based cost,
    # replace the calculation of 'term1' by the following commented-out lines.
    #
    # Note 1: Remember to change the definition of ``output_delta()`` below
    # Note 2: Remember to set activation function at ``__init__``, to the
    #         logistic function
    # Note 3: Notice that the "log" operation may underflow quite easily when a
    #         large number of examples (such as in M-NIST) are into play. If
    #         that happens, the cost may be set to +Inf and the minimization
    #         will stop abruptly. Using the MSE is a way to overcome this.

    # ML logistic cost term
    #h = self.forward(X)
    #logh = numpy.nan_to_num(numpy.log(h))
    #log1h = numpy.nan_to_num(numpy.log(1-h))
    #term1 = -(y*logh + ((1-y)*log1h)).sum() / y.shape[1]

    # MSE term
    term1 = 0.5 * numpy.sum( (self.forward(X)-y)**2 ) / y.shape[1]

    # Regularization term
    term2 = (0.5 * self.regularization * (numpy.sum(self.w1**2) + numpy.sum(self.w2**2)) / y.shape[1])

    return term1 + term2

  def output_delta(self, output, y):
    """Returns the expected delta on the output layer (i.e. the derivate of the
    cost w.r.t. the activations on the last layer) if one uses the MSE cost as
    declared above."""

    # If you would like to ues the "Logistic Maximum Likelihood"-based cost,
    # replace the calculation of the return value by the following commented
    # out line. Please read the Notes section on ``J()`` above.

    # Note: This is the "naive" implementation of the output delta. In these
    #       settings we calculate delta here and then multiply that by the
    #       activation_prime() (which is (output * (1 - output)!) on the
    #       backward() implementation. Technically, this should cancel out.
    #       Nevertheless, if you find problems concerning this division on your
    #       training, I recommend you simplify the problem by removing the term
    #       (output * (1 - output)) here and slightly modifying backward() on
    #       the evaluation of ``self.delta2`` to take that into account,
    #       replacing the line that says:
    #
    #       self.delta2 = self.b2 * activation_prime(self.z2)
    #
    #       by:
    #
    #       self.delta2 = self.b2

    # ML logistic cost delta
    #return ((output - y) / (output * (1 - output))) / y.shape[1]

    # Use this for the MSE cost **or** ML logistic cost with the simplification
    # noted above
    return (output - y) / y.shape[1]

  def forward(self, X):
    """Executes the forward step of a 2-layer neural network.

    Remember that:

    1. z = w^T . X

    and

    2. Output: a = g(z), with g being the logistic function

    Keyword attributes:

    X
      The input vector containing examples organized in columns. The input
      matrix does **not** contain the bias term.

    Returns the outputs of the network for each row in X. Accumulates hidden
    layer outputs and activations (for backward step). At the end of this
    procedure:

    self.a0
      Input, including the bias term for the hidden layer. 1 example per
      column. Bias = first row.

    self.z1
      Activations for every input X on hidden layer. z1 = w1^T * a0

    self.a1
      Output of the hidden layer, including the bias term for the output layer.
      1 example per column. Bias = first row. a1 = [1, act(z1)]

    self.z2
      Activations for the output layer. z2 = w2^T * a1.

    self.a2
      Outputs for the output layer. a2 = act(z2)

    Tip: You must first calculate the output of the first layer and then use
    that output to calculate the output of the second layer. It is not possible
    to do that at the same time.
    """

    self.a0 = numpy.vstack([numpy.ones(X.shape[1], dtype=float), X])
    self.z1 = numpy.dot(self.w1.T, self.a0)
    self.a1 = self.activation(self.z1)
    self.a1 = numpy.vstack([numpy.ones(self.a1.shape[1], dtype=float), self.a1])
    self.z2 = numpy.dot(self.w2.T, self.a1)
    self.a2 = self.activation(self.z2)

    return self.a2

  def backward(self, y):
    """Executes the backward step for training.

    In this phase, we calculate the error on the output layer and then use
    back-propagation to estimate the error on the hidden layer. We then use
    this estimated error to calculate the differences between what the layer
    output and the expected value.

    Keyword attributes:

    y
      The expected output for the last layer of the network. This is a simple 1D
      numpy.ndarray containing 1 value for each input example in the original
      input vector X.

    Sets the internal values for various variables:

    self.delta2
      Delta (derivative of J w.r.t. the activation on the last layer) for the
      output neurons:
      self.delta2_1 = (a2_1 - y) / (a2_1 * (1-a2_1)) / y.shape[1]
      N.B.: This is only valid if J is the ML logistic cost

    self.b2
      That is the back-propagated activation values, passing the delta values
      through the derivative of the activation function, w.r.t.  the previously
      calculated activation value: self.b2 = self.delta2 * activation'(z2) =
      self.delta2 * (1 - a2**2) [if the activation function is tanh].

      N.B.: This is not a matrix multiplication, but an element by element
      multiplication as delta2 and a2 are single column vectors.

    self.delta1
      Delta (error) for the hidden layer. This calculated back-propagating the
      b's from the output layer back the neuron. In this specific case:
      self.delta1 = w2 * self.b2

      N.B.: This is a matrix multiplication

    self.b1
      Back-propagated activation values for hidden neurons. The analogy is the
      same: self.b1 = self.delta1 * activation'(z1) = self.delta1 * (1 - a1**2)

      N.B.: This is not a matrix multiplication, but an element by element
      multiplication as delta1 and a1 are single column vectors.

    self.d1, self.d2

      The updates for each synapse are simply the multiplication of the a's and
      b's on each end. One important remark to get this computation right: one
      must generate a weight change matrix that is of the same size as the
      weight matrix. If that is not the case, something is wrong on the logic

      self.dL = self.a(L-1) * self.b(L).T / number-of-examples

      N.B.: This **is** a matrix multiplication, despite a and b are vectors.
    """

    # For the next part of this exercise, you will complete the calculation of
    # the deltas and the weight updates (d). Before you start filling this,
    # make sure you scan the code for forward() so that you understand which
    # variables are already preset to you.

    # Evaluate deltas, b's and d's. In doubt, look at the comments above
    self.delta2 = self.output_delta(self.a2, y)
    self.b2 = self.delta2 * self.activation_prime(self.a2)
    self.delta1 = numpy.dot(self.w2[1:], self.b2)
    self.b1 = self.delta1 * self.activation_prime(self.a1[1:])
    self.d2 = numpy.dot(self.a1, self.b2.T) / y.shape[1]
    self.d2 += (self.regularization / y.shape[1]) * self.w2
    self.d1 = numpy.dot(self.a0, self.b1.T) / y.shape[1]
    self.d1 += (self.regularization / y.shape[1]) * self.w1

  def CER(self, X, y):
    """Calculates the Classification Error Rate, a function of the weights of
    the network.

      CER = count ( round(MLP(X)) != y ) / X.shape[1]
    """

    est_cls = self.forward(X).argmax(axis=0)
    cls = y.argmax(axis=0)

    return sum( cls != est_cls ) / float(X.shape[1])

  def __str__(self):
    """Printer-friendly method"""

    retval = 'w1:\n' + str(self.w1) + '\n'
    retval += 'w2:\n' + str(self.w2)
    return retval

class Trainer:
  """Trains an MLP machine using LBFGS-B"""

  def __init__(self, seed, nhidden, regularization, grad_norm):
    """Initializes and loads the data

    seed
      An randomization seed (machine is initialized with it).

    nhidden
      Number of units (neurons) on the hidden layer.

    regularization
      A regularization parameter to be passed to the MLP.

    grad_norm
      The norm of the projected gradient. Training with LBFGS-B will stop when the surface respects this degree of "flatness".
    """

    self.seed = seed
    self.nhidden = nhidden
    self.regularization = regularization
    self.grad_norm = grad_norm

  def J(self, theta, machine, X, y):
    """
    Calculates the vectorized cost *J*, by unrolling the theta vectors into the
    network weights in an analogous way you did for the training.
    """

    machine.w1 = theta[:machine.w1.size].reshape(machine.w1.shape)
    machine.w2 = theta[machine.w1.size:].reshape(machine.w2.shape)
    return machine.J(X, y)

  def dJ(self, theta, machine, X, y):
    """
    Calculates the vectorized partial derivative of the cost *J* w.r.t. to
    **all** :math:`\theta`'s. Use the training dataset.
    """
    machine.w1 = theta[:machine.w1.size].reshape(machine.w1.shape)
    machine.w2 = theta[machine.w1.size:].reshape(machine.w2.shape)
    machine.forward(X)
    machine.backward(y)
    return numpy.hstack([machine.d1.flatten(), machine.d2.flatten()])

  def train(self, X, y):
    """
    Optimizes the machine parameters to fit the input data, using
    ``scipy.optimize.fmin_l_bfgs_b``.

    Keyword parameters:

    X
      The input data matrix. This must be a numpy.ndarray with 2 dimensions.
      Every column corresponds to one example of the data set, every row, one
      different feature. The input data vector X must not have the bias column
      added. It must be pre-normalized if necessary.

    y
      The expected output data matrix.

    Returns a trained machine.
    """

    # prepare the machine

    # Initialize the seed like you did on the previous exercise
    numpy.random.seed(self.seed)

    machine = Machine(self.nhidden, self.regularization)

    print 'Settings:'
    print '  * cost (J) = %g' % (machine.J(X, y),)
    cer = machine.CER(X, y)
    print('  * CER      = %g%% (%d sample(s))' % (100*cer, X.shape[1]*cer))
    print 'Training using scipy.optimize.fmin_l_bfgs_b()...'

    # theta0 is w1 and w2, flattened
    theta0 = numpy.hstack([machine.w1.flatten(), machine.w2.flatten()])

    # Fill in the right parameters so that the minimization can take place
    theta, cost, d = scipy.optimize.fmin_l_bfgs_b(
        self.J,
        theta0,
        self.dJ,
        (machine, X, y),
        pgtol=self.grad_norm,
        iprint=0,
        disp=2,
        )

    if d['warnflag'] == 0:

      print("** LBFGS converged successfuly **")
      machine.theta = theta
      print 'Final settings:'
      print '  * cost (J) = %g' % (cost,)
      #print '  * |cost\'(J)| = %s' % numpy.linalg.norm(d['grad'])
      cer = machine.CER(X, y)
      print('  * CER      = %g%% (%d sample(s))' % (100*cer, X.shape[1]*cer))
      return machine

    else:
      print("LBFGS did **not** converged:")
      if d['warnflag'] == 1:
        print("  Too many function evaluations")
      elif d['warnflag'] == 2:
        print("  %s" % d['task'])
      return None
