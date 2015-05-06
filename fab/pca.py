#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Wed 15 May 2013 15:27:57 CEST

"""In this exercise, you will implement yourself Principal Component Analysis
so that the program `ex1.py' works flawlessly. You should try to use a
vectorized solution in all your answers.
"""

import bob.learn.linear
import numpy

def pca_covmat(X):
  """Principal components of a given data matrix X using the Covariance Matrix.

  This method computes the principal components for a given data matrix X using
  its covariance matrix. The principal components correspond the direction of
  the data in which its points are maximally spread.

  As you saw in class, computing these principal components is equivalent to
  computing the eigen vectors U for the covariance matrix Sigma extracted from
  the data. The covariance matrix for the data is computed using the equation
  bellow::

    Sigma = mean((X-mu_X)(X-mu_X)^T)

  Once you have Sigma, it suffices to compute the eigen vectors U, solving the
  linear equation:

    (Sigma - e * I) * U = 0

  We will use numpy's LAPACK bindings to compute the solution to this linear
  equation, but you can use any other library. Because you know Sigma is
  symmetric, you better use ``numpy.linalg.eigh`` instead of
  ``numpy.linalg.eig``, because it is faster.

  Keyword arguments:

  X
    The data matrix in which each column corresponds to one observation of the
    data and each row, to one feature.

  You must return e, the eigen values and U, the eigen vectors of the
  covariance matrix Sigma, computed from the data. U must be arranged such that
  each column represents one eigen vector. For example, to project X on U we
  would do: X' = U^T X.
  """
  # Replace e and U by your calculations. Read docs above.
  # The shapes of e and U should be similar to the shapes below


  # (a) lecture-notes based approach
#  mu = X.mean(1);
#  COV = numpy.zeros((X.shape[0], X.shape[0]))
#  for i in range(0, X.shape[1]):
#    X_ = numpy.matrix(X[:,i]-mu).T
#    COV = COV + numpy.dot(X_, X_.T)
#  COV = COV / X.shape[1]  

  # (b) direct approach 
  X_ = X-numpy.kron(numpy.ones((X.shape[1],1)), X.mean(1)).T
  COV = numpy.dot(X_, X_.T) / (X.shape[1]-1)
  
  # get eigenvalues and -vectors and sort them
  e, U = numpy.linalg.eigh(COV)
  IdxSortedE = e.argsort();
  
  e = e[IdxSortedE];
  U = U[IdxSortedE,:]
  
#  e = numpy.ones((X.shape[0],), dtype=float)
#  U = numpy.ones((X.shape[0], X.shape[0]), dtype=float)

  return e, U

def pca_svd(X):
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

  # demean X first
  X_ = X-numpy.kron(numpy.ones((X.shape[1],1)), X.mean(1)).T
  
  # perform SVD
  U, s, V = numpy.linalg.svd(X_, 'false')

  # get eigenvalues
  e = (s*s) / (X.shape[1]-1)
  
  #import pdb; pdb.set_trace()
  
  return e, U

def pca_bob(X):
  """Calculates the PCA decomposition using Bob's builtin C++ code

  This method does the same as pca_svd() above, but is implemented using Bob's
  infrastructured. The returned machine knows how to load and save itself from
  HDF5 files and the eigen vectors are already organized by decreasing energy.

  It is here so that you see an example - you don't have to do anything for
  this part of the code.
  """

  pca_trainer = bob.learn.linear.PCATrainer()
  pca_machine, e = pca_trainer.train(X.T)
  return e, pca_machine.weights

def project(X, U_reduced):
  """Projects the data over the reduced version of U (eigen vectors of

  This function receives the input data for a given digit and creates a
  "projected" version of the digit using the reduced space chosen by yourself,
  with the ``--components`` command line option. The maths behind this
  operation are simple:

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

  # Replace the following line by your solution:
  X_prime = numpy.dot(U_reduced.T, X)

  return X_prime

def cosine_similarity(p, q):
  """Calculates the cosine similarity between to data vectors 'p' and 'q'.

  The cosine similarity between two vectors is defined as the cosine of the
  angle between these two vectors. This measure is therefore invariant to 'p'
  and 'q''s norms.

  N.B.: max(cos(x)) = 1, when x = 0 radians. So, similarity and distance play
  inverse roles for classification.

  You can calculate the cosine of the angle between two vectors, by dividing
  the dot-product of the two vectors by their multiplied L2-norms.

  s(p,q) = p q / |p||q|

  Where

  |p| = sqrt ( sum (p_i^2) )

  Tip: You can use ``numpy.linalg.norm`` to compute the normalization factors
  """

  # Replace the following line by your solution
  return (numpy.dot(p,q))/(numpy.linalg.norm(p) * numpy.linalg.norm(q))
