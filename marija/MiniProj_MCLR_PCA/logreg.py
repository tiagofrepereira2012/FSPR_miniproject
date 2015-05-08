#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Thu 21 Mar 2013 15:03:35 CET

"""Exercise 2: complete the missing bits in the functions bellow, so the
program ``ex2.py`` works flawlessly. See more details and test cases at our
course slides.

N.B.: You should try to use the vectorized form for all answers.
"""

import bob.io.base
import numpy
import scipy.optimize

def make_polynom(X, degree=1):
  """
  Returns the polynomial expansion for all variables in X.

  This method is implemented by observing that polynomial expansion can be
  treated as a recursive problem. For example, the polynomial expansion of 2
  variables (x1, x2) in degree 3 would be:

  .. math::

    P3 = hstack(P2, (x1^3, x2^3, x1^2*x2, x2^2*x1, x2^3))
    P2 = hstack(P1, (x1^2, x1*x2, x2^2))
    P1 = (x1, x2)

  The terms that need to be calculated at each level correspond to the
  combinations with replacement of variables, in groups ``degree``. For
  example:

  .. math::

    P3 = hstack(P2, (x1*x1*x1, x1*x1*x2, x1*x2*x2, x2*x2*x2))

  By using combinatorics and numpy matrix arrangements, we can easily produce
  the new terms at each recursion. By combining the terms, we get a quite
  vectorized version of the make_polynom() method.

  Parameters:

  X
    A 2D numpy.ndarray containing all observations for the input variable. This
    matrix contains 4 variables in this order (columns): sepal length, sepal
    width, petal length and petal width.

  degree
    The degree of the polynomial

  Returns a 2D numpy.ndarray with as many columns as polynom terms you would
  like to have.
  """
  from itertools import combinations_with_replacement

  retval = []
  for P in combinations_with_replacement(range(X.shape[1]), degree):
    retval.append(numpy.prod(X.T[P,:], axis=0).reshape(X.shape[0],1))
  if degree > 1: retval.insert(0, make_polynom(X, degree-1))
  return numpy.hstack(retval)

def add_bias(X):
  """Adds a bias term to the data matrix X"""
  return numpy.hstack([numpy.ones((len(X),1)), X])

class Machine:
  """A class to handle all run-time aspects for Logistic Regression"""

  def __init__(self, theta, norm=None):
    """Initalizes this machine with a set of theta parameters.

    Keyword parameters:

    theta
      A set of parameters for the Logistic Regression model. This must be an
      iterable (or numpy.ndarray) with all parameters for the model, including
      the bias term, which must be on entry 0 (the first entry at the
      iterable).

    norm
      Normalization parameters, a tuple with two components: the mean and the
      standard deviation. Each component must be an iterable (or numpy.ndarray)
      with a normalization factor for all terms. For the bias term (entry 0),
      set the mean to 0.0 and the standard deviation to 1.0.
    """

    self.theta = numpy.array(theta).copy()

    if norm is not None:
      self.norm  = (
          numpy.array(norm[0]).copy(),
          numpy.array(norm[1]).copy(),
          )

      # check
      if self.norm[0].shape != self.norm[1].shape:
        raise RuntimeError, "Normalization parameters differ in shape"

      if self.norm[0].shape != self.theta.shape:
        raise RuntimeError, "Normalization parameters and theta differ in shape"

    else:
      self.norm = (
          numpy.zeros(self.theta.shape),
          numpy.ones(self.theta.shape),
          )

  def set_norm(self, X):
    """Sets the normalization parameters for this machine from a dataset.

    Keyword parameters:

    X
      The input data matrix. This must be a numpy.ndarray with 2 dimensions.
      Every row corresponds to one example of the data set, every column, one
      different feature. The input data is expected to have the bias column
      added.

    Returns a tuple containing the normalization parameters (average and
    standard deviation) for the input dataset.
    """

    self.norm = (
        numpy.zeros(self.theta.shape),
        numpy.ones(self.theta.shape),
        )

    # sets all terms, but the bias
    self.norm[0][1:] = numpy.mean(X[:,1:], axis=0)
    self.norm[1][1:] = numpy.std(X[:,1:], axis=0, ddof=1)

  def normalize(self, X):
    """Normalizes the given dataset

    X
      The input data matrix. This must be a numpy.ndarray with 2 dimensions.
      Every row corresponds to one example of the data set, every column, one
      different feature. The input data is expected to have the bias column
      added.

    Returns the normalized dataset.
    """
    return (X - self.norm[0]) / self.norm[1]

  def __call__(self, X, pre_normed=False):
    """Spits out the hypothesis given the data.

    Keyword parameters:

    X
      The input data matrix. This must be a numpy.ndarray with 2 dimensions.
      Every row corresponds to one example of the data set, every column, one
      different feature. The input data vector X must already have the bias
      column added (first column).

    pre_normed
      if pre_norm is set, it means X is already normalized.
    """
    Xnorm = X if pre_normed else self.normalize(X)
    return 1. / (1. + numpy.exp(-numpy.dot(Xnorm, self.theta)))

  def J(self, X, y, regularizer=0.0, pre_normed=False):
    """
    Calculates the logistic regression cost

    Keyword parameters:

    X
      The input data matrix. This must be a numpy.ndarray with 2 dimensions.
      Every row corresponds to one example of the data set, every column, one
      different feature. The input data vector X must already have the bias
      column added (first column).

    y
      The expected output data matrix.

    regularizer
      A regularization parameter

    pre_normed
      if pre_norm is set, it means X is already normalized.
    """

    Xnorm = X if pre_normed else self.normalize(X)
    h = self(Xnorm, pre_normed=True)
    logh = numpy.nan_to_num(numpy.log(h))
    log1h = numpy.nan_to_num(numpy.log(1-h))

    # TODO: program the regularization term:
    regularization_term = regularizer*(self.theta[1:]**2).sum()

    main_term = -(y*logh + ((1-y)*log1h)).mean()

    return main_term + regularization_term

  def dJ(self, X, y, regularizer=0.0, pre_normed=False):
    """
    Calculates the logistic regression first derivative of the cost

    Keyword parameters:

    X
      The input data matrix. This must be a numpy.ndarray with 2 dimensions.
      Every row corresponds to one example of the data set, every column, one
      different feature. The input data vector X must already have the bias
      column added (first column).

    y
      The expected output data matrix.

    regularizer
      A regularization parameter

    pre_normed
      if pre_norm is set, it means X is already normalized.
    """

    Xnorm = X if pre_normed else self.normalize(X)
    retval = ((self(Xnorm, pre_normed=True) - y) * Xnorm.T).T.mean(axis=0)

    # TODO: Add the regularization term correction
    retval[1:]+=regularizer*(self.theta[1:])/len(Xnorm)
    return retval

  def CER(self, X, y, pre_normed=False):
    """
    Calculates the (vectorized) classification error rate for a 2-class
    problem.

    Keyword parameters:

    X
      The input data matrix. This must be a numpy.ndarray with 2 dimensions.
      Every row corresponds to one example of the data set, every column, one
      different feature. The input data vector X must already have the bias
      column added (first column).

    y
      The expected output data matrix.

    pre_normed
      if pre_norm is set, it means X is already normalized.
    """

    Xnorm = X if pre_normed else self.normalize(X)
    h = self(Xnorm, pre_normed=True)
    h[h<0.5] = 0.0
    h[h>=0.5] = 1.0
    errors = (h != y).sum()
    return float(errors)/len(Xnorm)

  def save(self, h5f):
    """Saves the machine to a pre-opened HDF5 file

    Keyword parameters

    h5f
      An object of type bob.io.base.HDF5File that has been opened for writing
      and pre-set so that this machine dumps its parameters on the expected
      location.
    """

    # Use bob.io.base.HDF5File to save: 'theta', norm[0] and norm[1] into a file
    h5f.set('theta', self.theta)
    h5f.set('subtract', self.norm[0])
    h5f.set('divide', self.norm[1])

  def load(self, h5f):
    """Loads the machine from a pre-opened HDF5 file

    Keyword parameters

    h5f
      An object of type bob.io.base.HDF5File that has been opened for reading
      and pre-set so that this machine reads its parameters from the expected
      location.
    """

    # Recover theta, norm[0] and norm[1] from the file
    self.theta = h5f.read('theta')
    self.norm = h5f.read('subtract'), h5f.read('divide')

class Trainer:
  """A class to handle all training aspects for Logistic Regression"""

  def __init__(self, normalize=False, regularizer=0.0):
    """Initializes and loads the data

    normalize
      True if we should normalize the dataset (after polynomial expansion).
    """

    self.normalize = normalize
    self.regularizer = regularizer

  def J(self, theta, machine, X, y):
    """
    Calculates the vectorized cost *J*.
    """
    machine.theta = theta
    return machine.J(X, y, self.regularizer)

  def dJ(self, theta, machine, X, y):
    """
    Calculates the vectorized partial derivative of the cost *J* w.r.t. to
    **all** :math:`\theta`'s. Use the training dataset.
    """
    machine.theta = theta
    return machine.dJ(X, y, self.regularizer)

  def train(self, X, y):
    """
    Optimizes the machine parameters to fit the input data, using
    ``scipy.optimize.fmin_l_bfgs_b``.

    Keyword parameters:

    X
      The input data matrix. This must be a numpy.ndarray with 2 dimensions.
      Every row corresponds to one example of the data set, every column, one
      different feature. The input data vector X must already have the bias
      column added (first column).

    y
      The expected output data matrix.

    Returns a trained machine.
    """

    # prepare the machine
    theta0 = numpy.zeros(X.shape[1])
    machine = Machine(theta0)
    if self.normalize:
      machine.set_norm(X)

    print 'Settings:'
    #print '  * initial guess = %s' % ([k for k in theta0],)
    print '  * cost (J) = %g' % (machine.J(X, y),)
    print '  * CER      = %g%%' % (100*machine.CER(X, y),)
    print 'Training using scipy.optimize.fmin_l_bfgs_b()...'

    # Fill in the right parameters so that the minimization can take place
    theta, cost, d = scipy.optimize.fmin_l_bfgs_b(
        self.J,
        theta0,
        self.dJ,
        (machine, X, y),
        )

    if d['warnflag'] == 0:

      print("** LBFGS converged successfuly **")
      machine.theta = theta
      print 'Final settings:'
      #print '  * theta = %s' % ([k for k in theta],)
      print '  * cost (J) = %g' % (cost,)
      print '  * CER      = %g%%' % (100*machine.CER(X, y),)
      return machine

    else:
      print("LBFGS did **not** converged:")
      if d['warnflag'] == 1:
        print("  Too many function evaluations")
      elif d['warnflag'] == 2:
        print("  %s" % d['task'])
      return None
