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

# My utility box
import data

__epilog__ = """\
  To run the evaluation with default parameters, type:

    $ %(prog)s mlp.hdf5

  To plot the confusion matrices, type:

    $ %(prog)s --plot mlp.hdf5

  Use --help to see other options.
""" % {'prog': os.path.basename(sys.argv[0])}

def main():

  parser = argparse.ArgumentParser(description=__doc__, epilog=__epilog__,
      formatter_class=argparse.RawDescriptionHelpFormatter)

  parser.add_argument('-t', '--test', action='store_true', default=False,
      help=argparse.SUPPRESS)

  parser.add_argument('-p', '--plot', action='store_true', default=False,
    help="Visualizes confusion matrices graphically (it is off by default)")

  parser.add_argument('machine_file', default='mlp.hdf5',
      metavar='PATH', help="Path to the filename where to store the trained machine (defaults to %(default)s)")

  args = parser.parse_args()

  # loads the machine file
  print("Loading Pen-digits training set...")
  X_train, labels_train = data.as_mnist('data/train.hdf5', 28)
  print("Loading Pen-digits development set...")
  X_devel, labels_devel = data.as_mnist('data/devel.hdf5', 28)

  # creates a matrix for the MLP output in which only the correct label
  # position is set with +1.
  y_train = -1*numpy.ones((10, len(labels_train)), dtype=float)
  y_train[labels_train, range(len(labels_train))] = 1.0
  y_devel = -1*numpy.ones((10, len(labels_devel)), dtype=float)
  y_devel[labels_devel, range(len(labels_devel))] = 1.0

  # Normalizing input set
  #print("Normalizing input data...")
  X_train  = X_train.astype(float)
  X_train /= 255.
  X_devel = X_devel.astype(float)
  X_devel /= 255.

  f = bob.io.base.HDF5File(args.machine_file, 'r')
  X_mean = f.read('X_mean')
  X_train -= X_mean
  X_devel -= X_mean

  import project as answers

  machine = answers.Machine()
  machine.load(f)
  del f

  hidden = machine.w2.shape[0] - 1
  print("Number of inputs               : %d" % (28*28,))
  print("Number of hidden units         : %d" % (hidden,))
  print("Number of outputs              : 10")
  print("Total number of free parameters: %d" % \
      ((28*28+1)*hidden+(hidden+1)*10))


  print("** FULL Train set results (%d examples):" % X_train.shape[1])
  print("  * cost (J) = %f" % machine.J(X_train, y_train))
  cer = machine.CER(X_train, y_train)
  print('  * CER      = %g%% (%d sample(s))' % (100*cer, X_train.shape[1]*cer))

  print("** Development set results (%d examples):" % X_devel.shape[1])
  print("  * cost (J) = %f" % machine.J(X_devel, y_devel))
  cer = machine.CER(X_devel, y_devel)
  print('  * CER      = %g%% (%d sample(s))' % (100*cer, X_devel.shape[1]*cer))

  if args.test:

    print("Loading Pen-digits (writer-dependent) test set...")
    X_test, labels_test = data.as_mnist('data/test-dependent.hdf5', 28)

    # creates a matrix for the MLP output in which only the correct label
    # position is set with +1.
    y_test = -1*numpy.ones((10, len(labels_test)), dtype=float)
    y_test[labels_test, range(len(labels_test))] = 1.0

    X_test = X_test.astype(float)
    X_test /= 255.
    X_test -= X_mean

    print("** Test set results (%d examples):" % X_test.shape[1])
    print("  * cost (J) = %f" % machine.J(X_test, y_test))
    cer = machine.CER(X_test, y_test)
    print('  * CER      = %g%% (%d sample(s))' % (100*cer, X_test.shape[1]*cer))

  if args.plot:
    print("Plotting confusion matrices...")

    # plot confusion matrix
    N = 2
    if args.test: N = 3
    fig = mpl.figure(figsize=(N*6, 6))

    def plot_cm(X, y, set_name):

      # plot training
      cm = data.confusion_matrix(y.argmax(axis=0), machine.forward(X).argmax(axis=0))
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

    mpl.subplot(1, N, 1)
    plot_cm(X_train, y_train, 'train')
    mpl.subplot(1, N, 2)
    plot_cm(X_devel, y_devel, 'devel.')

    if args.test:
      mpl.subplot(1, N, 3)
      plot_cm(X_test, y_test, 'test')

    print("Close the plot window to terminate.")
    mpl.show()

if __name__ == '__main__':
  main()
