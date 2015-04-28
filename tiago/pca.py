#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Wed 15 May 2013 15:27:57 CEST

"""In this exercise, you will implement yourself Principal Component Analysis
so that the program `ex1.py' works flawlessly. You should try to use a
vectorized solution in all your answers.
"""

import numpy
import matplotlib.pyplot as mpl

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


  def plot(self):

    total_energy = numpy.sum(self.e)
    print("Plotting energy load curve...")

    mpl.plot(range(len(self.e)), 100*numpy.cumsum(self.e)/total_energy)
    mpl.title('Energy loading curve for M-NIST (training set)')
    mpl.xlabel('Number of components')
    mpl.ylabel('Energy (percentage)')
    mpl.grid()
    print("Close the plot window to continue.")
    mpl.show()


  def save(self, hdf5file):
    hdf5file.create_group("pca")
    hdf5file.cd("pca")
    hdf5file.set('U',self.U)
    hdf5file.set('e',self.e)
    hdf5file.set('components',self.components)
    hdf5file.cd("..")


  def load(self, hdf5file):
    hdf5file.cd("pca")
    self.U = hdf5file.read('U')
    self.e = hdf5file.read('e')
    self.components = hdf5file.read('components')
    hdf5file.cd("..")


class Trainer:
  """
  Trains a PCA using SVD
  """

  def train(self, X, components=0):
    """Calculates components of a given data matrix X using Singular Value
    Decomposition.

    This method computes the principal components for a given data matrix X using
    SVD. You gain using this technique because you never have to compute Sigma,
    risking to loose numerical precision.

    Singular value decomposition is factorization of a matrix M, with m rows and
    n columns, such that:

      M = U S V*

      U : unitary matrix of size m x m - a.k.a., left singular vectors of M
      S : rectangular diagonal matrix with nonnegative real numbers, size m x n
      V*: (the conjugate transpose of V) unitary matrix of size n x n, right
          singular vectors of M

    (see http://en.wikipedia.org/wiki/Singular_value_decomposition)

    We can use this property to avoid the computation of the covariance matrix of
    X, if we note the following:

      X = U S V*, so
      XX' = U S V* V S U*
      XX' = U S^2 U*

    This means that the U matrix obtained by SVD contains the eigen vectors of
    the original data matrix X and S corresponds to the square root of the eigen
    values.

    N.B.: So that U corresponds to principal components of X, we must also
    remove the means of X before applying SVD.

    We will use numpy's LAPACK bindings to compute the solution to this linear
    equation, but you can use any other library. Note that you don't need to
    calculate the full-ranks U and V matrices and with that you can save time.
    When using ``numpy.linalg.svd``, make sure that you set ``full_matrices`` to
    ``False``.

    Keyword arguments:

    X
      The data matrix in which each column corresponds to one observation of the
      data and each row, to one feature.

    You must return e, the eigen values and U, the eigen vectors of the
    covariance matrix Sigma, computed from the data. U must be arranged such that
    each column represents one eigen vector. For example, to project X on U we
    would do: X' = U^T X.
    """

    U, sigma, V = numpy.linalg.svd(X-numpy.mean(X,axis=1).reshape(X.shape[0],1), full_matrices=False)
    #e,_ = numpy.linalg.eig(numpy.cov(X))

    N = X.shape[1]
    e = (sigma**2)/(N-1)

    machine = Machine(components=components)
    machine.e = e
    machine.U = U

    return machine



