#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Thu 23 Apr 13:09:15 2015 CEST

"""Read and provide functions to visualise the pen digits"""

import sys
import numpy
import matplotlib.pyplot as mpl
import bob.io.base

def load(filename):
    """Loads a given data set in the available data

    Parameters:

      filename (str): The filename to read data from.


    Returns:

      list: A python list of lists. Elements in the list correspond to the
        different digit classes, from 0 to 9, in order. Each element in the
        list contains a single ``numpy.ndarray`` with the points for each of
        the examples in that set.
    """

    f = bob.io.base.HDF5File(filename)
    retval = []
    for k in range(10):
        p = f.read('%d/points' % k)
        s = f.read('%d/sizes' % k)
        cs = s.cumsum()
        cs2 = cs.copy()
        cs2[-1] = 0
        ranges = zip(numpy.roll(cs2, 1), cs)
        retval.append([p[r[0]:r[1]] for r in ranges])

    return retval


def as_mnist(filename, imwidth):
  """Loads the input data in a graphical representation, like the one for the
  M-NIST example (lab3, ex3).

  Parameters:

    filename (str): The filename to read data from.

  Returns:

    images (numpy.ndarray): A 2D numpy.ndarray with as many columns as examples
      in the dataset, as many rows as pixels (actually, there are 28x28 = 784
      rows). The pixels of each image are unrolled in C-scan order (i.e., first
      row 0, then row 1, etc.).

    labels (numpy.ndarray): A 1D numpy.ndarray with as many elements as
      examples in the dataset. Each element contains one of the 10 labels
      (0..9)
  """

  images = []
  labels = []

  for cls, data in enumerate(load(filename)):
      for example in data:
          labels.append(cls)
          image = numpy.zeros(shape=(imwidth, imwidth), dtype='uint8')
          for (x, y) in example:
              x_ = int(round(imwidth * x))
              y_ = int(round(1-(imwidth * y)))
              image[y_, x_] = 255
          images.append(image.flatten())

  return numpy.vstack(images).T.copy(), numpy.array(labels)


def confusion_matrix(expected, predicted):
  """Transforms the prediction list into a confusion matrix

  This method takes lists of expected and predicted classes and returns a
  confusion matrix, which represents the percentage of classified examples in
  each combination of "expected class" of samples and "predicated class" of the
  same samples.

  Parameters:

    expected (numpy.ndarray, 1D): The ground-thruth
    predicted (numpy.ndarray, 1D): The predicted classes with your neural
      network

  You must combine these scores column wise and determine what are the
  annotated rates (below) for each of the column entries, returning the
  following 2D numpy.ndarray::

    [ TP0    / N0    FP1(0) / N1    FP2(0) / N2 ... ]
    [ FP0(1) / N0    TP1    / N1    FP2(1) / N2 ... ]
    [ FP0(2) / N0    FP1(2) / N1    TP2    / N2 ... ]
    [     ...              ...            ...       ]
    [ FP0(9) / N0    FP1(9) / N1    TP9    / N9 ... ]

  Where:

  TPx / Nx
    True Positive Rate for class ``x``

  FPx(y) / Nz
    Rate of False Positives for class ``y``, from class ``x``. That is,
    elements from class ``x`` that have been **incorrectly** classified as
    ``y``.
  """

  retval = numpy.zeros((10,10), dtype=float)

  for k in range(10):
    pred_k = predicted[expected==k] # predictions that are supposed to be 'k'
    retval[:,k] = numpy.array([len(pred_k[pred_k==p]) for p in range(10)])
    retval[:,k] /= len(pred_k)

  return retval


def plot_numbers(X, labels, examples, imwidth):
  """Visualizes a grid of numbers picked randomly from the available samples.

  There are ten samples per line. Each line corresponds to a single sample type
  (digit).
  """

  plotting_image = numpy.zeros((imwidth*10,imwidth*examples), dtype='uint8')
  for y in range(10):
    digits = X[:,labels==y].T
    for x, image in enumerate(digits[numpy.random.randint(0,len(digits),(examples,))]):
      plotting_image[y*imwidth:(y+1)*imwidth, x*imwidth:(x+1)*imwidth] = image.reshape(imwidth, imwidth)

  mpl.imshow(plotting_image, cmap=mpl.cm.Greys)
  mpl.axis('off')
  mpl.title('Pen-digits Examples')

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


  mu_X=numpy.mean(X, axis=1)


  Sigma=((X-numpy.matrix([mu_X,]*X.shape[1]).transpose())*(X-numpy.matrix([mu_X,]*X.shape[1]).transpose()).transpose())/(X.shape[1]-1)


  # Replace e and U by your calculations. Read docs above.
  # The shapes of e and U should be similar to the shapes below

  ee,UU=numpy.linalg.eigh(Sigma)

  e=numpy.sort(ee)[::-1]
  tt=numpy.argsort(ee)[::-1]
  U=UU[:,tt]
  


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


  mu_X=numpy.mean(X, axis=1)
  t_X=X-numpy.matrix([mu_X,]*X.shape[1]).transpose()
  U, e, V=numpy.linalg.svd(t_X,full_matrices=False)
  return (e**2)/(X.shape[1]-1), U


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

  
  X_prime=numpy.dot(U_reduced.transpose(),X)
  

  return X_prime

def lda(X, y):
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
  
  # Replace e and U by your calculations. Read docs above.
  # The shapes of e and U should be similar to the shapes below
  #e = numpy.ones((X.shape[0],), dtype=float)
  #U = numpy.ones((X.shape[0], X.shape[0]), dtype=float)

  # Step 1: compute the class means mu_c, starting from the sum_c
  # Tip: Place the results in the matrix mu_c in which each column represents
  #      the mean for one class.

  mu_c=numpy.array([numpy.mean(X[:,y==k], axis=1)for k in range(10)]).T

  # Step 1.5: computes the number of elements in each class

  n_c=numpy.array([sum(y==k)for k in range (10)])

  # Step 2: computes the global mean mu
  
  
  mu=numpy.sum(mu_c*n_c, axis=1)/X.shape[1]

  # Step 3: compute the between-class scatter Sb
  mu_c_mu=(mu_c.T-mu).T
  Sb=numpy.dot(n_c*mu_c_mu, mu_c_mu.T)

  # Step 4: compute the within-class scatter Sw
  
  Sw=numpy.zeros((X.shape[0], X.shape[0]), dtype=float)
  for k in range(10):
    X_c_mu_c=(X[:,y==k].T-mu_c[:,k]).T
    Sw+=numpy.dot(X_c_mu_c, X_c_mu_c.T)

  # Step 5: calculate the eigen vectors and values of Sw^-1 Sb
  
 

  e, U = numpy.linalg.eig(numpy.dot(numpy.linalg.pinv(Sw),Sb))
  

  return e, U



if __name__ == '__main__':

    imwidth = 28
    images, labels = as_mnist('data/train.hdf5', imwidth)
    plot_numbers(images, labels, 10, imwidth)
    print("Close the plot window to terminate.")
    mpl.show()
    sys.exit(0)
