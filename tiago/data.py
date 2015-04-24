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


if __name__ == '__main__':

    imwidth = 28
    images, labels = as_mnist('data/train.hdf5', imwidth)
    plot_numbers(images, labels, 10, imwidth)
    print("Close the plot window to terminate.")
    mpl.show()
    sys.exit(0)
