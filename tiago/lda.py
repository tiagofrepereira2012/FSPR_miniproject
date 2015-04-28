#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Wed 15 May 2013 15:27:57 CEST

"""In this exercise, you will implement yourself Linear Discriminant Analysis
so that the program `ex2.py' works flawlessly. You should try to use a
vectorized solution in all your answers.
"""

import numpy
import utils

class Machine:
  """
  Represents a machine for Linear Projection
  """

  def __init__(self, components=0):

    self.components = components

    self.U = None
    self.e = None


  #def __init__(self, hdf5file="lda.hdf5"):
    #self.load(hdf5file)


  def project(self, X):
    """Projects the data over the reduced version of U (eigen vectors of

    X' = U^t X

    Keyword arguments:

    X
      The input matrix (already) rasterized in C-scan order, so that every
      column corresponds to one example of the data.

    U_reduced
      The reduced set of eigen vectors where only the first C eigen vectors have
      been kept following the command line options.

    Returns the projected matrix X' such that X' = U^t X
    """

    if self.U is None:
      raise ValueError("The machine was not trained")

    # Replace the following line by your solution:
    X_proj = numpy.dot(self.U[:,0:self.components].T,X)

    return X_proj


  def creating_models(self, X,labels, n_classes=10):

    models = numpy.ndarray((self.components, n_classes), dtype=float)
    for k in range(n_classes):
      X_proj = self.project(X)
      models[:,k] = numpy.mean(X_proj[:, labels == k], axis=1)

    return models


  def predict(self, X, models, similatiry_measure, arg_func):

    X_proj = self.project(X)

    predicted_labels = []
    for i in range(X_proj.shape[1]):
      similarities = []
      for j in range(models.shape[1]):
        similarities.append(similatiry_measure(X_proj[:,i],models[:,j]))
      predicted_labels.append(arg_func(similarities))

    return numpy.array(predicted_labels)


  def save(self, hdf5file):
    hdf5file.create_group("lda")
    hdf5file.cd("lda")
    hdf5file.set('U',self.U)
    hdf5file.set('e',self.e)
    hdf5file.set('components',self.components)
    hdf5file.cd("..")


  def load(self, hdf5file):
    hdf5file.cd("lda")
    self.U = hdf5file.read('U')
    self.e = hdf5file.read('e')
    self.components = hdf5file.read('components')
    hdf5file.cd("..")



class Trainer:
  """
  Trains an LDA
  """

  def train(self, X, y, components):
    """Calculates the projection matrix U to perform LDA on X with labels y.

    LDA finds the projecting matrix W that allows us to linearly project X to
    another (sub) space in which the between-class and within-class variances are
    jointly optimized: the between-class variance is maximized while the
    with-class is minimized. The (inverse) cost function for this criteria can be
    posed as the following:
  
      J(W) = W^T Sb W / W^T Sw W
  
    Where:

      W - the transformation matrix that converts X into the LD space
      Sb - the between-class scatter; it has dimensions (X.shape[0], X.shape[0])
      Sw - the within-class; it also has dimensions (X.shape[0], X.shape[0])

    N.B.: A scatter matrix equals the covariance matrix if we remove the division
    factor.

    Because this cost function is convex, you can just find its maximum by
    solving dJ/dW = 0. As exposed in class, this problem can be re-formulated as
    finding the eigen values (l_i) that solve the following condition:

      Sb = l_i Sw, or

      (Sb - l_i Sw) = 0

    If you can do that, then the respective eigen vectors that correspond to the
    eigen values l_i form W.

    To calculate Sb and Sw, you start by computing the class means, for each
    class c:

      mu_c = mean(X when y == c)
   
    Then the overall mean:

      mu = mean(n_c * mu_c)

    Sb is easy, it is the scatter matrix considering mu_c and mu

      Sb = sum_C ( n_c (mu_c - mu)(mu_c - mu)^T )

    Sw looks the same, but we compute the scatter over each individual component
    for each class and sum all up:

      Sw = sum_C ( sum_n_c (x_i^c - mu_c) * (x_i^c - mu_c)^T )

    You can call ``numpy.linalg.eig`` to solve the problem (Sb - l_i Sw).

    Keyword arguments:

    X
      The data matrix in which each column corresponds to one observation of the
      data and each row, to one feature.

    y
      The label vector containing the labels of each column of X

    TODO: You must return e, the eigen values and U, the eigen vectors of the
    generalized Eigen decomposition for LDA, computed from the data. U must be
    arranged such that each column represents one eigen vector. For example, to
    project X on U we would do: X' = U^T X.
    """


    def sort_eigenvectors(e, U):
      """Sorts eigen vectors based on the eigen values (reverse order)"""
      indexes, e = zip(*sorted(enumerate(e), key=lambda x: x[1], reverse=True))
      return numpy.array(e), U[:,indexes]


    # Replace e and U by your calculations. Read docs above.
    # The shapes of e and U should be similar to the shapes below
    #e = numpy.ones((X.shape[0],), dtype=float)
    #U = numpy.ones((X.shape[0], X.shape[0]), dtype=float)

    # Step 1: compute the class means mu_c, starting from the sum_c
    # Tip: Place the results in the matrix mu_c in which each column represents
    #      the mean for one class.

    n_classes = 10#len(set(y))

    #mu_c = numpy.zeros(shape=(n_classes,X.shape[1]))
    mu_c = numpy.zeros(shape=(n_classes,X.shape[0]))
    n_c = numpy.zeros(shape=(n_classes,))
    for i in range(n_classes):
      mu_c[i,:] = numpy.mean(X[:,y==i],axis=1)
      n_c[i]    = X[:,y==i].shape[1]
      #mu_c[i,:] = numpy.mean(X[y==i,:],axis=0)
      #n_c[i]    = X[y==i,:].shape[0]

    #import ipdb; ipdb.set_trace();
    # Step 1.5: computes the number of elements in each class
    #n_c = numpy.zeros(shape=(10,))
    #for i in range(10):
      #n_c[i] = X[:,y==i].shape[1]


    # Step 2: computes the global mean mu
    # Tip: Test what happens when you do matrix * vector in numpy - maybe you
    #      can re-use mu_c and n_c?
    #mu = numpy.mean(X,axis=0)
    mu = numpy.mean(X,axis=1)

    # Step 3: compute the between-class scatter Sb
    # Tip: Sb is a square matrix with as many rows/columns as rows in X
    #      (i.e. 784 == X.shape[0])
    # Tip: You can use numpy.dot to calculate Sb in a single shot, if you can
    #      first calculate (mu_c - mu).
    # Tip: Use numpy's operation broadcasting to vectorize the subtraction of the
    #      various class means in mu_c with mu.
    #Sb = numpy.zeros((X.shape[1], X.shape[1]), dtype=float)
    Sb = numpy.zeros((X.shape[0], X.shape[0]), dtype=float)
    muc_minus_mu = mu_c - mu
    Sb = numpy.dot((n_c.reshape(n_classes,1)*muc_minus_mu).T,muc_minus_mu)
  

    # Step 4: compute the within-class scatter Sw
    # Tip: Here you will need a for loop
    #Sw = numpy.zeros((X.shape[1], X.shape[1]), dtype=float)
    Sw = numpy.zeros((X.shape[0], X.shape[0]), dtype=float)
    for i in range(y.shape[0]):
      #X_c = X[i,:] - mu_c[y[i],:]
      X_c = X[:,i] - mu_c[y[i],:]
      #X_c = X_c.reshape(X.shape[1],1)
      X_c = X_c.reshape(X.shape[0],1)
      Sw += numpy.dot(X_c,X_c.T)


    # Step 5: calculate the eigen vectors and values of Sw^-1 Sb
    # Tip: Because we use raw pixels, the matrix Sw is probably singular
    #      (determinant == 0). Use ``numpy.linalg.pinv(Sw)`` to compute the
    #      pseudo-inverse instead of ``numpy.linalg.inv``.
    # Tip: Use ``numpy.linalg.eig`` (since you know Sw^-1 Sb may not be symmetric)

    #print numpy.linalg.matrix_rank(Sb)
    #print numpy.linalg.matrix_rank(Sw)

    Sw_inv = numpy.linalg.pinv(Sw)
    Sw_inv_Sb = numpy.dot(Sw_inv, Sb)

    e, U = numpy.linalg.eig(Sw_inv_Sb)
    e, U  = sort_eigenvectors(e,U)

    machine = Machine(components=components)
    machine.e = e
    machine.U = U

    return machine
