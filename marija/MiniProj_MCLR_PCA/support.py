#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Mon 25 Mar 2013 14:47:09 CET

"""Exercise 3: complete the missing bits in the functions bellow, so the
program ``ex3.py`` works flawlessly. See more details and test cases at our
course slides.
"""

import numpy

def make_subset(y, cls):
  """Creates a subset of data to train a particular LR machine.

  Keyword parameters:

  y
    A 1D numpy.ndarray containing the class of the Iris flower. The mapping
    goes like this:

    0
      setosa

    1
      virginica

    2
      versicolor

  cls
    The class that will be subset.

  You must return a new (replacement) vector for y that contains 1's and 0's
  instead of the class number, depending on the selected class. For example, if
  the selected class is 2, then the returned new vector must have all positions
  set to 0, except when y == 2.

  Tip: Use advanced indexing for achieving this transformation.
  """

  retval = numpy.zeros(len(y))
  retval[y == cls] = 1.
  return retval

def predict_class(scores):
  """Predicts the class of every row of X given a list of machines

  This method returns a numpy.ndarray containing the indexes of the machines
  that satisfy the following condition:

  .. math::

    class = \arg \max_k h_{\bm{\theta}(k)} = \arg \max_k P(y=k | \mathbf{x}; \bm{\theta}(k)) with $k=1..3$

  Keyword parameters:

  scores
    A 2D numpy.ndarray containing the scores for a given data set, obtained by
    each machine in the pool. Each row corresponds to one example and each
    column, to the output of the machine with class corresponding to the column
    index. For example, at column 0, we have the scores for the machine that
    was trained for class 0.

  Your task is to return a 1D numpy.ndarray in which every entry corresponds to
  a colum in scores. The values of this array should match the index of the
  score column (i.e., the machine class). For example, if ``retval[0]`` is 2,
  it means the entry in ``scores[0]`` is predicted as the class 2.

  Tip: Compute argmax along the rows (using ``argmax(axis=1)``).
  """

  return scores.argmax(axis=1)

def confusion_matrix(scores):
  """Transforms the prediction list into a confusion matrix

  This method takes lists of score lists and returns a confusion matrix, which
  represents the percentage of classified examples in each combination of
  "expected class" of samples and "predicated class" of the same samples.

  Keyword parameters:

  scores
    A list of lists, each with 3 entries with 1D numpy.ndarray's corresponding
    to the following combinations of machines and (real) data classes::

    [ machine0(data0)   machine0(data1)   machine0(data2) ]  |
    [ machine1(data0)   machine1(data1)   machine1(data2) ]  |
    [ machine2(data0)   machine2(data1)   machine2(data2) ]  |
                                                             v
    ----> real class                                       predicted class

  You must combine these scores column wise and determine what are the
  annotated rates (below) for each of the column entries, returning the
  following 2D numpy.ndarray::

    [ TP0    / N0    FP1(0) / N1    FP2(0) / N2 ]
    [ FP0(1) / N0    TP1    / N1    FP2(1) / N2 ]
    [ FP0(2) / N0    FP1(2) / N1    TP2    / N2 ]

  Where:

  TPx / Nx
    True Positive Rate for class ``x``

  FPx(y) / Nz
    Rate of False Positives for class ``y``, from class ``x``. That is,
    elements from class ``x`` that have been **incorrectly** classified as
    ``y``.

  To do so, you must combine the scores of each column and determine what is
  the class (using ``argmax``) for each sample. Once that is determined, you
  must count the rates above forming each cell of the return value.
  """

  retval = numpy.ones((10,10), dtype='float64')

  for real in range(10):
    col_scores = numpy.vstack([k[real] for k in scores]).T
    predictions = predict_class(col_scores) #predictions for class 'real'
    for predicted_as in range(10):
      retval[predicted_as, real] = len(predictions[predictions == predicted_as]) / float(len(predictions))

  return retval
