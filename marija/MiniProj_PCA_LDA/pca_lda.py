#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Wed 24 Apr 09:20:37 2013

"""Evaluation routine for exercise 3.
"""

import os
import sys
import argparse
import numpy
import matplotlib.pyplot as mpl
import bob.io.base
import ipdb
# My utility box
import data

__epilog__ = """\
  To run the evaluation with default parameters, type:

    $ %(prog)s mlp.hdf5

  To plot the confusion matrices, type:

    $ %(prog)s --plot mlp.hdf5

  Use --help to see other options.
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

def euclidean_distance(p, q):
  """Calculates the euclidean distance between two normalised data vectors 'p'
  and 'q'.
  
  The euclidean distance between two normalised vectors is defined as the
  **scalar**:

  d(p',q') = sqrt ( sum (p'_i - q'_i)^2 )

  Where

  t' = t / |t|, and

  |t| = sqrt ( sum (t_i^2) )

  Tip: You can use ``numpy.linalg.norm`` (see help) to compute the euclidean
  distance and the normalization factors.
  """

  return numpy.linalg.norm(p/numpy.linalg.norm(p) - q/numpy.linalg.norm(q))

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

  return numpy.dot(p,q)/(numpy.linalg.norm(p)*numpy.linalg.norm(q))

def sort_eigenvectors(e, U):
  """Sorts eigen vectors based on the eigen values (reverse order)"""
  indexes, e = zip(*sorted(enumerate(e), key=lambda x: x[1], reverse=True))
  return numpy.array(e), U[:,indexes]

def plot_cm(y_exp, y_pred, set_name):

      # plot confusion matrix
      cm = data.confusion_matrix(y_exp, y_pred)
      res = mpl.imshow(cm, cmap=mpl.cm.summer, interpolation='nearest')

      for x in numpy.arange(cm.shape[0]):
        for y in numpy.arange(cm.shape[1]):
          col = 'white'
          if cm[x,y] > 0.5: col = 'black'
          mpl.annotate('%.2f' % (100*cm[x,y],), xy=(y,x), color=col,
              fontsize=8, horizontalalignment='center', verticalalignment='center')

      classes = [str(k) for k in range(10)]

      mpl.xticks(numpy.arange(10), classes)
      mpl.yticks(numpy.arange(10), classes, rotation=90)
      mpl.ylabel("(Your prediction)")
      mpl.xlabel("(Real class)")
      mpl.title("Confusion Matrix (%s set) - in %%" % set_name)

def CER(y, y_pred):
    
    return sum( y_pred!= y) / float(y.shape[0])

def main():

  parser = argparse.ArgumentParser(description=__doc__, epilog=__epilog__,
      formatter_class=argparse.RawDescriptionHelpFormatter)

  parser.add_argument('-t', '--test', action='store_true', default=False,
      help=argparse.SUPPRESS)
  
  parser.add_argument('-c', '--components', default=300, type=int,
      help="Number of principal components to keep (defaults to %(default)s)")
 
  parser.add_argument('-l', '--lda', default=5, type=int,
      help="Number of lda components to keep (defaults to %(default)s)")

  parser.add_argument('-p', '--plot', action='store_true', default=False,
    help="Visualizes confusion matrices graphically (it is off by default)")

  parser.add_argument('-e', '--energy', action='store_true', default=False,
    help="Visualizes energy load curve graphically (it is off by default)")

  args = parser.parse_args()

  # loads data files
  print("Loading Pen-digits training set...")
  X_train, y_train = data.as_mnist('data/train.hdf5', 28)
  print("Loading Pen-digits development set...")
  X_devel, y_devel = data.as_mnist('data/devel.hdf5', 28)

  # Normalizing input set
  print("Normalizing input data...")
  X_train  = X_train.astype(float)
  X_train /= 255.
  X_devel = X_devel.astype(float)
  X_devel /= 255.

  X_mean   = X_train.mean(axis=1).reshape(-1,1)
  X_train -= X_mean
  X_devel -= X_mean

  import pca as answers1

  e, U_pca = answers1.pca_covmat(X_train)
  e, U_pca = sort_eigenvectors(e, U_pca)

  total_energy = sum(e)
  if args.energy:
    print("Plotting energy load curve...")
    mpl.plot(range(len(e)),
        100*numpy.cumsum(e)/total_energy)
    mpl.title('Energy loading curve (training set)')
    mpl.xlabel('Number of components')
    mpl.ylabel('Energy (percentage)')
    mpl.grid()
    print("Close the plot window to continue.")
    mpl.show()

  print("With %d components (your choice), you preserve %.2f%% of the energy" % (args.components, 100*sum(e  [:args.components])/total_energy)) 

  # PCA
  print("Projecting train set using %d PCA components..." % args.components)
  X_train_pca = numpy.dot(U_pca[:,:args.components].T, X_train)

  print("Projecting development set using %d PCA components..." % args.components)
  X_devel_pca = numpy.dot(U_pca[:,:args.components].T, X_devel)
  
  
  import lda as answers2
  
  e, U = answers2.lda(X_train_pca, y_train)
  e, U = sort_eigenvectors(e, U)
  
  
  
  # Keep args.lda components
  U = U[:, :args.lda]
  
  # LDA
  print("Projecting train set using %d LDA components..." % args.lda)
  X_train_proj = numpy.dot(U.T, X_train_pca)
  print("Projecting development set using %d LDA components..." % args.lda)
  X_devel_proj = numpy.dot(U.T, X_devel_pca)

  models = numpy.ndarray((args.lda, 10), dtype=float)
  
  print("Creating average models for each digit...")
  for k in range(10):
    models[:,k] = numpy.mean(X_train_proj[:, y_train == k], axis=1)
  print("Evaluating performance using cosine similarity (training set)...")
  y_train_est_cosine = numpy.argmax([numpy.apply_along_axis(cosine_similarity, 0, X_train_proj, k) for k in models.T], axis=0)
  cosine_accuracy = sum(y_train_est_cosine == y_train)/float(len(y_train))
  print("Training set accuracy = %.2f %% (over %d samples)" % \
      (100*cosine_accuracy, len(y_train)))

  cer = CER( y_train, y_train_est_cosine)
  print('  * CER (train)      = %g%% (%d sample(s))' % (100*cer, X_train_proj.shape[1]*cer))

  # Evaluate performance on the development set
  

  print("Evaluating performance using cosine similarity (development set)...")
  y_devel_est_cosine = numpy.argmax([numpy.apply_along_axis(cosine_similarity, 0, X_devel_proj, k) for k in models.T], axis=0)
  cosine_accuracy = sum(y_devel_est_cosine == y_devel)/float(len(y_devel))
  print("Development set accuracy = %.2f %% (over %d samples)" % \
      (100*cosine_accuracy, len(y_devel)))
      
  cer = CER( y_devel, y_devel_est_cosine)
  print('  * CER (devel)      = %g%% (%d sample(s))' % (100*cer, X_devel_proj.shape[1]*cer))
  


  if args.test:
    print("Loading Pen-digits (writer-dependent) test set...")
    X_test, y_test = data.as_mnist('data/test-dependent.hdf5', 28)
  
    X_test = X_test.astype(float)
    X_test /= 255.
    X_test -= X_mean

    # PCA
    print("Projecting test set using %d PCA components..." % args.components)
    X_test_pca = numpy.dot(U_pca[:,:args.components].T, X_test)

    # LDA
    print("Projecting test set using %d LDA components..." % args.lda)
    X_test_proj = numpy.dot(U.T, X_test_pca)

    # Evaluate performance on the test set
    print("Evaluating performance using cosine similarity (test set)...")
    y_test_est_cosine = numpy.argmax([numpy.apply_along_axis(cosine_similarity, 0, X_test_proj, k) for k in models.T], axis=0)
    #ipdb.set_trace()
    cosine_accuracy = sum(y_test_est_cosine == y_test)/float(len(y_test))
    print("Test set accuracy = %.2f %% (over %d samples)" % \
        (100*cosine_accuracy, len(y_test)))

    cer = CER( y_test, y_test_est_cosine)
    print('  * CER (test)      = %g%% (%d sample(s))' % (100*cer, X_test_proj.shape[1]*cer))
  
  if args.plot:
    # plot confusion matrix
    N=3
    fig = mpl.figure(figsize=(N*6, 6))
    mpl.subplot(1, N, 1)
    plot_cm(y_train, y_train_est_cosine, 'train')
    mpl.subplot(1, N, 2)
    plot_cm(y_devel, y_devel_est_cosine, 'devel')
    mpl.subplot(1, N, 3)
    plot_cm(y_test, y_test_est_cosine, 'test')
    mpl.show() 

if __name__ == '__main__':
  main()
