#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Sat 20 Apr 15:17:52 2013 CEST

"""Exercise 3: Vectorized Neural Networks for Pen-digit classification

In this exercise you will implement the hyperbolic tangent activation and MSE
cost function on neural networks, and see that in action with the Pen-digit
dataset.
"""

import os
import sys
import argparse
import numpy
import time
import matplotlib.pyplot as mpl
import bob.io.base

# My utility box
import data

__epilog__ = """\
  To run the problem with default parameters, type:

    $ %(prog)s mlp.hdf5

  Use --help to see other options.
""" % {'prog': os.path.basename(sys.argv[0])}


def main():

  parser = argparse.ArgumentParser(description=__doc__, epilog=__epilog__,
      formatter_class=argparse.RawDescriptionHelpFormatter)

  parser.add_argument('-s', '--seed', type=int, default=0,
      metavar='SEED (INT)', help="Random (initialization) seed (defaults to %(default)s)")

  parser.add_argument('-H', '--hidden', type=int, default=10,
      metavar='INT', help="Number of hidden units (defaults to %(default)s)")

  parser.add_argument('-l', '--regularization', type=float, default=0.,
      metavar='FLOAT', help="Regularization parameter (defaults to %(default)s - i.e. no reguralization)")

  parser.add_argument('-n', '--projected-gradient-norm', type=float,
      default=1e-6, metavar='FLOAT', help='The norm of the projected gradient.  Training with LBFGS-B will stop when the surface respects this degree of "flatness" (defaults to %(default)s)')

  parser.add_argument('-a', '--pca', type=float,
      default=0.,  help='Perform the PCA to the training data. The defaults values is %(default)s for the energy, which means NO PCA.')

  parser.add_argument('-p', '--plot', action='store_true', default=False,
    help="So far, visualizes the PCA energy load curve!")


  parser.add_argument('machine_file', default='mlp.hdf5',
      metavar='MACHINE', help="Path to the filename where to store the trained machine (defaults to %(default)s)")

  args = parser.parse_args()

  ##### START of program

  print("Pen-digit Classification using an MLP with tanh activation")
  if args.pca > 0:
    n_inputs = args.pca
  else:
    n_inputs = 28 * 28

  print("Number of inputs               : %d" % (n_inputs,))
  print("Number of hidden units         : %d" % (args.hidden,))
  print("Number of outputs              : 10")
  print("Total number of free parameters: %d" % \
      ((n_inputs+1)*args.hidden+(args.hidden+1)*10))


  print("Loading Pen-digit training set...")
  X_train, labels_train = data.as_mnist('data/train.hdf5', 28)

  print("Loading Pen-digit development set...")
  X_devel, labels_devel = data.as_mnist('data/devel.hdf5', 28)

  # creates a matrix for the MLP output in which only the correct label
  # position is set with +1.
  y_train = -1*numpy.ones((10, len(labels_train)), dtype=float)
  y_train[labels_train, range(len(labels_train))] = 1.0
  y_devel = -1*numpy.ones((10, len(labels_devel)), dtype=float)
  y_devel[labels_devel, range(len(labels_devel))] = 1.0

  print("Using %d samples for training..." % len(y_train.T))

  # Normalizing input set
  #print("Normalizing input data...")
  X_train  = X_train.astype(float)

  X_train /= 255.
  X_mean   = X_train.mean(axis=1).reshape(-1,1)
  #X_std   = X_train.std(axis=1, ddof=1).reshape(-1,1)
  #X_train = (X_train - X_mean) / X_std
  X_train -= X_mean

  X_devel = X_devel.astype(float)
  X_devel /= 255.
  X_devel -= X_mean

  f = bob.io.base.HDF5File(args.machine_file, 'w')
  f.set('X_mean', X_mean)

  #Check if will train the PCA
  if args.pca > 0.:
    import pca

    pca_trainer = pca.Trainer()
    pca_machine = pca_trainer.train(X_train, args.pca) 
    pca_machine.save(f)
    

    #e_pca,U_pca = do_pca(X_train, args.plot)
    X_train = pca_machine.project(X_train)
    X_devel = pca_machine.project(X_devel)

  import project as answers

  trainer = answers.Trainer(args.seed, args.hidden, args.regularization,
      args.projected_gradient_norm)
  start = time.time()

  machine = trainer.train(X_train, y_train*0.8)
  total = time.time() - start

  if machine is None:
    print("Training did **NOT** finish. Aborting...")
    sys.exit(1)

  sys.stdout.write("** Training is over, took %.2f minute(s)\n" % (total/60.))
  sys.stdout.flush()

  machine.save(f)
  #del f


  print("** Development set results (%d examples):" % X_devel.shape[1])
  print("  * cost (J) = %f" % machine.J(X_devel, y_devel))
  cer = machine.CER(X_devel, y_devel)
  print('  * CER      = %g%% (%d sample(s))' % (100*cer, X_devel.shape[1]*cer))

if __name__ == '__main__':
  main()
