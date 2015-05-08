#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Mon 25 Mar 2013 16:07:33 CET

"""Exercise 3: Multiclass, Multivariate logistic regression using built-in
optimization.

In this exercise you will try to predict the Iris Flower class using LLR with
L-BFGS. You will use all variables available for your prediction.
"""

import os
import sys
import argparse
import numpy
import bob.io.base
import bob.db.iris
import matplotlib.pyplot as mpl
import data as nums
import ipdb

__epilog__ = """\
  To evaluate your multi-class classifier:

  $ %(prog)s setosa.hdf5 versicolor.hdf5 virginica.hdf5
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

  classes = ['zero','one', 'two', 'three','four','five','six','seven','eight','nine']

  parser = argparse.ArgumentParser(description=__doc__, epilog=__epilog__,
      formatter_class=argparse.RawDescriptionHelpFormatter)

  parser.add_argument('-p', '--plot', action='store_true', default=False,
    help="Visualizes confusion matrices graphically (it is off by default)")

  parser.add_argument('input', metavar='HDF5', type=str, nargs=10,
      help="The names of 10 HDF5 files containing the parameters for each of the following  LR machines: zero, one,...,nine")

    
  parser.add_argument('-c', '--components', default=5, type=int,
      help="Number of principal components to keep (defaults to %(default)s)")

  parser.add_argument('--test', action='store_true', default=False, help=argparse.SUPPRESS)

  args = parser.parse_args()
  args.machines = dict(
      zero=args.input[0],
      one=args.input[1],
      two=args.input[2],
      three=args.input[3],
      four=args.input[4],
      five=args.input[5],
      six=args.input[6],
      seven=args.input[7],
      eight=args.input[8], 
      nine=args.input[9],   
      )
  
  args.subset = ['train', 'devel']
  if args.test: args.subset.append('test')

  ##### START of program
  import support as answers
  import logreg

  
  print("Loading Pen-digits training set...")
  X_train, y_train = nums.as_mnist('data/train.hdf5', 28)
  print("Loading Pen-digits development set...")
  X_devel, y_devel = nums.as_mnist('data/devel.hdf5', 28)
  print("Loading Pen-digits test set...")
  X_test, y_test = nums.as_mnist('data/test-dependent.hdf5', 28)
  
  

  # Normalizing input set
  #print("Normalizing input data...")
  X_train  = X_train.astype(float)
  X_train /= 255.
  X_devel = X_devel.astype(float)
  X_devel /= 255.
  X_test = X_test.astype(float)
  X_test /= 255.
  

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

  

  print("Loading user machines...")
  machines = {}
  for key in args.machines:

    m = logreg.Machine(numpy.zeros((1,1)))
    f = bob.io.base.HDF5File(args.machines[key], 'r')
    m.load(f)
    # load supplemental material
   
    m.poly_degree = f.get_attribute('polynomial_degree')
    machines[key] = m
    del f
    train = {}
    devel = {}
    test  = {}

    print("Doing polynomial expansion of degree %d for machine `%s'..." % \
        (m.poly_degree, key))
    lengths = [0, 0, 0]

    for key in classes:
      
      train[key] = logreg.add_bias(logreg.make_polynom(X_train[y_train==classes.index(key),:], m.poly_degree))
      lengths[0] += len(train[key])
      devel[key]   = logreg.add_bias(logreg.make_polynom(X_devel[y_devel==classes.index(key),:], m.poly_degree))
      lengths[1] += len(devel[key])
      test[key]  = logreg.add_bias(logreg.make_polynom(X_test[y_test==classes.index(key),:], m.poly_degree))
      lengths[2] += len(test[key])
      #ipdb.set_trace()
    m.data = dict(train=train, devel=devel, test=test)
    
  #ipdb.set_trace()
  # evaluate machines on all 3 sets:
  ordered_machines = [machines[k] for k in classes]
  cm = {} # confusion matrix
  cer_subset={}
  #ipdb.set_trace()
  for subset in args.subset:

    scores = []

    # scan over each machine and then over the real class of data
    for mach in classes:
      m = machines[mach] #our current machine
      data = m.data[subset] #dict(setosa=..., versicolor=..., virginica=...)
      #ipdb.set_trace() 
      row_scores = []
      for real_cls in classes: row_scores.append(m(data[real_cls]))
      scores.append(row_scores)

    
    cm[subset] = answers.confusion_matrix(scores)
    #print "Confusion matrix for %s set:" % subset
    #print cm[subset]

    l_total=0
    l_false=0
    for real in range(10):
        col_scores = numpy.vstack([k[real] for k in scores]).T
        predictions = answers.predict_class(col_scores) #predictions for class 'real'
        l_total += len(predictions) 
        l_false += len(predictions[predictions != real])
        
    	
    cer_subset[subset] = float(l_false)/float(l_total)
    print("**Results for %s set:" % subset)
    print('  * CER      = %g%% )' % (100*cer_subset[subset]))
    #ipdb.set_trace()

  if args.plot:
    # plot confusion matrix
    N = 2
    if args.test: N = 3
    fig = mpl.figure(figsize=(N*6, 6))
    for subset in args.subset:
      mpl.subplot(1, N, args.subset.index(subset)+1)
      res = mpl.imshow(cm[subset], cmap=mpl.cm.summer, interpolation='nearest')

      for x in numpy.arange(cm[subset].shape[0]):
        for y in numpy.arange(cm[subset].shape[1]):
          col = 'white'
          if cm[subset][x,y] > 0.5: col = 'black'
          mpl.annotate('%.2f' % (100*cm[subset][x,y],), xy=(y,x), color=col,
              horizontalalignment='center', verticalalignment='center')

      mpl.xticks(numpy.arange(10), ['0', '1', '2', '3', '4', '5', '6', '7','8', '9'])
      mpl.yticks(numpy.arange(10), ['0', '1', '2', '3', '4', '5', '6', '7','8', '9'], rotation=90)
      mpl.ylabel("(Your prediction)")
      mpl.xlabel("(Real class)")
      mpl.title("Confusion Matrix (%s set)" % subset)

    print("Close the plot window to terminate.")
    mpl.show()

if __name__ == '__main__':
  main()
