#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Wed 24 Apr 09:22:56 2013 

"""Utilities for exercise 3, shared accross different scripts.
"""

import os
import numpy
import struct
import gzip
import matplotlib.pyplot as mpl

def check_gradient(machine):
  """Runs a simple gradient checking procedure on the MLP machine

  Keyword arguments:

  machine
    The input machine to test

  Suppose you have a function f(x) that purportedly computes dJ(x)/dx; you'd
  like to check if f is outputting correct derivative values.

  Imagine a random value for x, called xt (for test). Now imagine you modify
  one of the entries in xt so that xt+ has that entry added with a small
  (positive) value E and xv- has the same entry subtracted.

  In this case, one can use a truncated Taylor expansion of the derivative
  to calculate the approximate supposed value:

  f(x) ~ ( J(xt+) - J(xt-) ) / ( 2 * E )

  The degree to which these two values should approximate each other will
  depend on the details of J. But assuming E = 10−4, you’ll usually find that
  the left- and right-hand sides of the above will agree to at least 4
  significant digits (and often many more).

  We generate random inputs to the neural net and display both the numerical
  approximation and what your machine computes.
  """

  xt_length = machine.w1.size + machine.w2.size
  xt = numpy.random.rand((xt_length))
  E = 1e-4

  # Generate some random input and a fixed output
  X = numpy.random.rand(machine.w1.shape[0]-1, 1)
  y = numpy.ones((machine.w2.shape[1],1), dtype=float)

  # Evaluate the numerical gradients
  numerical = numpy.zeros(xt.shape, dtype=float)
  for k in range(xt_length):
    xt_plus = xt.copy()
    xt_plus[k] += E
    machine.w1 = xt_plus[:machine.w1.size].reshape(machine.w1.shape)
    machine.w2 = xt_plus[machine.w1.size:].reshape(machine.w2.shape)
    J_plus = machine.J(X, y)
    
    xt_minus = xt.copy()
    xt_minus[k] -= E
    machine.w1 = xt_minus[:machine.w1.size].reshape(machine.w1.shape)
    machine.w2 = xt_minus[machine.w1.size:].reshape(machine.w2.shape)
    J_minus = machine.J(X, y)

    numerical[k] = (J_plus - J_minus) / (2 * E)
   
  # Evaluate the gradient following your machine
  machine.w1 = xt[:machine.w1.size].reshape(machine.w1.shape)
  machine.w2 = xt[machine.w1.size:].reshape(machine.w2.shape)
  machine.forward(X)
  machine.backward(y)
  yours = numpy.hstack([machine.d1.flatten(), machine.d2.flatten()])

  # Print, in a comparative manner
  print " Flat # | Numerical       | Yours"
  print "--------+-----------------|------------------"
  for k in range(len(yours)):
    print "   %2d   | %.6e   | %.6e" % (k, numerical[k], yours[k])

def load_set(train):
  """Loads the M-NIST data set located on directory 'mnist'.
  
  Keyword parameters:

  train (bool)
    If set to ``True``, load the training set, otherwise, loads the test set.

  Returns a tuple composed of images and labels as 2D numpy arrays organized as
  follows:

  images
    A 2D numpy.ndarray with as many columns as examples in the dataset, as many
    rows as pixels (actually, there are 28x28 = 784 rows). The pixels of each
    image are unrolled in C-scan order (i.e., first row 0, then row 1, etc.).

  labels
    A 2D numpy.ndarray with as many columns as examples in the dataset, and 10
    rows. Each row corresponds to the expected output of the MLP network using
    one-hot-encoding.
  """

  path = 'mnist'

  if train:
    fname_images = os.path.join(path, 'train-images-idx3-ubyte.gz')
    fname_labels = os.path.join(path, 'train-labels-idx1-ubyte.gz')
  else:
    fname_images = os.path.join(path, 't10k-images-idx3-ubyte.gz')
    fname_labels = os.path.join(path, 't10k-labels-idx1-ubyte.gz')

  with gzip.GzipFile(fname_labels, 'rb') as f:
    # reads 2 big-ending integers
    magic_nr, n_examples = struct.unpack(">II", f.read(8))
    # reads the rest, using an uint8 dataformat (endian-less)
    labels = numpy.fromstring(f.read(), dtype='uint8')

  with gzip.GzipFile(fname_images, 'rb') as f:
    # reads 4 big-ending integers
    magic_nr, n_examples, rows, cols = struct.unpack(">IIII", f.read(16))
    shape = (n_examples, rows*cols)
    # reads the rest, using an uint8 dataformat (endian-less)
    images = numpy.fromstring(f.read(), dtype='uint8').reshape(shape).T

  return images, labels

def confusion_matrix(expected, predicted):
  """Transforms the prediction list into a confusion matrix
  
  This method takes lists of expected and predicted classes and returns a
  confusion matrix, which represents the percentage of classified examples in
  each combination of "expected class" of samples and "predicated class" of the
  same samples.

  Keyword parameters:

  expected (numpy.ndarray, 1D)
    The ground-thruth

  predicted (numpy.ndarray, 1D)
    The predicted classes with your neural network

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

def plot_numbers(X, labels, examples):
  """Visualizes a grid of numbers picked randomly from the available samples.

  There are ten samples per line. Each line corresponds to a single sample type
  (digit).
  """

  imwidth = 28

  plotting_image = numpy.zeros((imwidth*10,imwidth*examples), dtype='uint8')
  for y in range(10):
    digits = X[:,labels==y].T
    for x, image in enumerate(digits[numpy.random.randint(0,len(digits),(examples,))]):
      plotting_image[y*imwidth:(y+1)*imwidth, x*imwidth:(x+1)*imwidth] = image.reshape(imwidth, imwidth)

  mpl.imshow(plotting_image, cmap=mpl.cm.Greys)
  mpl.axis('off')
  mpl.title('M-NIST Example Digits')
