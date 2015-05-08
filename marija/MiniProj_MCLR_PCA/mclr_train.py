#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Mon 11 Mar 2013 11:00:24 CET

"""Exercise 3: Multiclass, Multivariate logistic regression using built-in
optimization.

In this exercise you will try to predict the Iris Flower class using LLR with
L-BFGS. You will use all variables available for your prediction.
"""

import os
import sys
import argparse
import bob.io.base
#import bob.db.iris
import numpy
import matplotlib.pyplot as mpl
import data
import ipdb
__epilog__ = """\
List of variables available:

  slen - Sepal length
  swid - Sepal width
  plen - Petal length
  pwid - Petal width

  To train to identify 'setosa':

  $ %(prog)s --variable plen swid -- setosa

  To visualize the decision surface (only if 2 variables chosen):

  $ %(prog)s --plot --variable plen swid -- setosa versicolor
""" % {'prog': os.path.basename(sys.argv[0])}

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

  return numpy.dot(U_reduced.t, X)

def sort_eigenvectors(e, U):
  """Sorts eigen vectors based on the eigen values (reverse order)"""
  indexes, e = zip(*sorted(enumerate(e), key=lambda x: x[1], reverse=True))
  return numpy.array(e), U[:,indexes]


def main():

  
  parser = argparse.ArgumentParser(description=__doc__, epilog=__epilog__,
      formatter_class=argparse.RawDescriptionHelpFormatter)

  
  parser.add_argument('-d', '--polynomial-degree', metavar='INT>0', type=int,
      default=1, dest='P', help="Polynomial degree to use for the regression (defaults to %(default)s)")

  parser.add_argument('-n', '--normalize', action='store_true', default=False,
      help="Turn-ON normalization (it is off by default)")

  parser.add_argument('-l', '--regularization-parameter', type=float,
      dest='L', metavar='FLOAT', default=0.0,
      help="Regularization parameter (defaults to %(default)s)")

  parser.add_argument('-o', '--output', type=str,
      help="Overides the output HDF5 model filename for the multiclass classifier (defaults to <flower-class>.hdf5)")

  parser.add_argument('-p', '--plot', action='store_true', default=False,
      help="Turn-ON plotting **after** training (it is off by default)")
  
  parser.add_argument('-c', '--components', default=5, type=int,
      help="Number of principal components to keep (defaults to %(default)s)")

  parser.add_argument('-x', '--axis', metavar='FLOAT', type=float, nargs=4,
      help="If set, allows you to control the decision boundary axis for plotting. You should pass 4 parameters: xmin, xmax, ymin, ymax")

  classes = ['zero','one', 'two', 'three','four','five','six','seven','eight','nine']

  parser.add_argument('cls', metavar='CLASS', type=str, choices = classes, help="The class you want to train the machine for. Choose from %s" % "|".join(classes))

  args = parser.parse_args()
 
  if args.output is None: args.output = '%s.hdf5' % args.cls

  ##### START of program

  print("Loading Pen-digits training set...")
  X_train, y_train = data.as_mnist('data/train.hdf5', 28)
  print("Loading Pen-digits development set...")
  X_devel, y_devel = data.as_mnist('data/devel.hdf5', 28)
  print("Loading Pen-digits test set...")
  X_test, y_test = data.as_mnist('data/test-dependent.hdf5', 28)

  # Normalizing input set
  #print("Normalizing input data...")
  X_train  = X_train.astype(float)
  X_train /= 255.
  X_devel = X_devel.astype(float)
  X_devel /= 255.
  X_test = X_test.astype(float)
  X_test /= 255.
  
  #ipdb.set_trace()
  # f = bob.io.base.HDF5File(args.machine_file, 'r')
  #X_mean = f.read('X_mean')
  X_mean   = X_train.mean(axis=1).reshape(-1,1)
  X_train -= X_mean
  X_devel -= X_mean
  X_test -= X_mean

  import pca as answers1

  e, U_pca = answers1.pca_covmat(X_train)
  e, U_pca = sort_eigenvectors(e, U_pca)

  X_train_pca = numpy.dot(U_pca[:,:args.components].T, X_train)
  X_devel_pca = numpy.dot(U_pca[:,:args.components].T, X_devel)
  X_test_pca = numpy.dot(U_pca[:,:args.components].T, X_test)

  X_train = X_train_pca.T
  X_devel = X_devel_pca.T
  X_test  = X_test_pca.T

  import support as answers
  import logreg

  trainer = logreg.Trainer(args.normalize, args.L)
  #ipdb.set_trace()
  y_train_sub = answers.make_subset(y_train, classes.index(args.cls))
  X_train_expanded = logreg.add_bias(logreg.make_polynom(X_train, args.P))

  #ipdb.set_trace()
  print("Training machine for `%s' flower class..." % args.cls)
  machine = trainer.train(X_train_expanded, y_train_sub)

  if machine is None: sys.exit(1)

  print("Development set statistics:")
  y_devel_sub = answers.make_subset(y_devel, classes.index(args.cls))
  X_devel_expanded = logreg.add_bias(logreg.make_polynom(X_devel, args.P))
  print '  * cost (J) = %g' % (machine.J(X_devel_expanded, y_devel_sub),)
  print '  * CER      = %g%%' % (100*machine.CER(X_devel_expanded, y_devel_sub),)

  print("Training is over. Storing machine to file `%s'..." % args.output)
  output = bob.io.base.HDF5File(args.output, 'w')
  output.set_attribute('polynomial_degree', args.P)
  machine.save(output)


if __name__ == '__main__':
  main()
