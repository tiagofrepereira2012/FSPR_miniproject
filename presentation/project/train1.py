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
import data1
from sklearn.mixture import GMM

__epilog__ = """\
  To run the problem with default parameters, type:

    $ %(prog)s mlp.hdf5

  Use --help to see other options.
""" % {'prog': os.path.basename(sys.argv[0])}

def sort_eigenvectors(e, U):
  """Sorts eigen vectors based on the eigen values (reverse order)"""
  indexes, e = zip(*sorted(enumerate(e), key=lambda x: x[1], reverse=True))
  return numpy.array(e), U[:,indexes]

def CER(X, y):
  """	Calculates the Classification Error Rate, a function of the weights of
    the network.
        X= expected labels
        y= predicted labels
   
      CER = count ( round(MLP(X)) != y ) / X.shape[1]
    """
  return sum(y!=X)/float(len(X))

def create_GMM(nb_classes, nb_com, covar_type, n_iteractions, X_proj_m, labels_m):
  #from sklearn.mixture import GMM
  #print X_proj_m.shape
  
  #print numpy.array(labels_m)==2
  #print (X_proj_m[numpy.array(labels_m)==2,:]).shape
  # creating the classifier
  classifier=[None]*nb_classes
  for i in range(nb_classes):
    classifier[i] =  GMM(n_components=nb_com, covariance_type=covar_type, n_iter=n_iteractions)
    classifier[i].fit(X_proj_m[numpy.array(labels_m)==i,:], numpy.array(labels_m)==i)

  return classifier

def GMM_classification(nb_classes, classifier, X_proj_m):
  prediction=numpy.ones((X_proj_m.shape[0],nb_classes))
  for i in range(nb_classes):
    prediction[:,i],tmp=(classifier[i].score_samples(X_proj_m))

  print prediction[1,:]
  return prediction


def main():

  parser = argparse.ArgumentParser(description=__doc__, epilog=__epilog__, formatter_class=argparse.RawDescriptionHelpFormatter)

  parser.add_argument('-p', '--plot', action='store_true', default=False, help="Turn-ON plotting **after** training (it is off by default)")

  parser.add_argument('-PCA', '--PCA', action='store_true', default=False, help="Dimensionality reduction with PCA (defaults to %(default)s)")

  parser.add_argument('-c', '--components', default=5, type=int, help="Number of principal components to keep (defaults to %(default)s)")

  parser.add_argument('-LDA', '--LDA', action='store_true', default=False, help="Dimensionality reduction with LDA (defaults to %(default)s)")
 
  parser.add_argument('-kNN', '--kNN', action='store_true', default=False, help="Classification using kNN method (defaults to %(default)s)")

  parser.add_argument('-nn', '--neighbors', default=9, type=int, help="Number of neighbors for kNN classification (defaults to %(default)s)")

  parser.add_argument('-GMM', '--GMM', action='store_true', default=False, help="Classification using GMM method (defaults to %(default)s)")

  parser.add_argument('-n_iter', '--n_iter', default=1000000, type=int, help="Number of EM iterations on GMM classification (defaults to %(default)s)")

  parser.add_argument('-cov_type', '--cov_type', default="full", type=str, help="String describing the type of covariance parameters on GMM classification. Must be one of 'spherical', 'tied', 'diag' or 'full' (defaults to %(default)s)")

  parser.add_argument('-nGMM', '--nGMM', action='store_true', default=False, help="Classification using GMM method (defaults to %(default)s)")

  parser.add_argument('-nb_com', '--nb_com', default=4, type=int, help="Number of gaussians for each class on GMM classification (defaults to %(default)s)")



  args = parser.parse_args()

  ##### START of program

  print("Pen-digit Classification using an MLP with tanh activation")
  print("Number of inputs               : %d" % (28*28,))
   
  
######## Loading the data ########

  print("Loading Pen-digit training set...")
  X_train, labels_train = data1.as_mnist('data/train.hdf5', 28)
  print("Loading Pen-digit development set...")
  X_devel, labels_devel = data1.as_mnist('data/devel.hdf5', 28)
  print("Loading Pen-digits (writer-dependent) test set...")
  X_test, labels_test = data1.as_mnist('data/test-dependent.hdf5', 28)

  # setting the default case in the proper format for classicication
  X_proj_m=numpy.matrix(X_train.T)
  labels_train_m=numpy.matrix(labels_train)
  labels_train_m=labels_train_m.transpose()
  labels_m=numpy.array(labels_train_m).reshape(-1,).tolist()

  devel=numpy.matrix(X_devel.T)
  test_d=numpy.matrix(X_test.T)



  if args.PCA:
########    PCA   ##########
    print("Dimensionality reduction with PCA:\n")
    print("Number of components: %d" % args.components)

    e,U=data1.pca_svd(X_train)
    e,U=sort_eigenvectors(e, U)
  
    print("Energy loading table:")
    print(" Comps. | Total energy captured")
    print(" -------+----------------------")
    total_energy = sum(e)
    for k in range(1,10):
      print("  %3d   |   %.2f%%" % (k, 100.*sum(e[:k])/total_energy))
    for k in range(10, 100, 10):
      print("  %3d   |   %.2f%%" % (k, 100.*sum(e[:k])/total_energy))
    for k in range(100, 784, 100):
      print("  %3d   |   %.2f%%" % (k, 100.*sum(e[:k])/total_energy))

 


    # project the data to new hyperspace
    X_projected=data1.project(U[:,:args.components], X_train)
    # setting the data and the labels to the proper format for classification
    X_proj_m=numpy.matrix(X_projected)
  


    labels_train_m=numpy.matrix(labels_train)
    labels_train_m=labels_train_m.transpose()
    labels_m=numpy.array(labels_train_m).reshape(-1,).tolist()

    # project the devel data to new hyperspace
    devel=data1.project(U[:,:args.components], X_devel)
    devel=numpy.matrix(devel);

  

    # project the testing data to new hyperspace
    test_d=data1.project(U[:,:args.components], X_test)
    test_d=numpy.matrix(test_d)

  
  if args.LDA:
  #####    LDA    ##########

    print("Dimensionality reduction with LDA:\n")
  #inmplementing LDA
    e,U=data1.lda(X_train,labels_train)
    e,U=sort_eigenvectors(e, U)

  # number of classes
    nb_classes=numpy.unique(labels_train).shape[0]
  
  
  
  
    
    print("Energy loading table:")
    print(" Comps. | Total energy captured")
    print(" -------+----------------------")
    total_energy = sum(e[:(nb_classes)])
    for k in range(1,(nb_classes)):
      print("  %3d   |   %.2f%%" % (k, 100.*sum(e[:k])/total_energy))
 
  

    # project the training data to new hyperspace
    X_projected=data1.project(abs(U[:,:(nb_classes-1)]), X_train)
    # setting the data and the labels to the proper format for classification
    X_proj_m=numpy.matrix(X_projected)
        
    labels_train_m=numpy.matrix(labels_train)
    labels_train_m=labels_train_m.transpose()
    labels_m=numpy.array(labels_train_m).reshape(-1,).tolist()

    # project the devel data to new hyperspace
    devel=data1.project(abs(U[:,:(nb_classes-1)]), X_devel)
    devel=numpy.matrix(devel);

  

    # project the devel data to new hyperspace
    test_d=data1.project(abs(U[:,:(nb_classes-1)]), X_test)
    test_d=numpy.matrix(test_d);

  
  
  
  if args.kNN:
  #####   kNN   ######
    print "Implementing kNN classification: \n"
    print("Number of neighbors: %d" % args.neighbors)
  # implementing kNN classification using sklearn module
    from sklearn.neighbors import KNeighborsClassifier
  # creating the classifier
    neigh = KNeighborsClassifier(n_neighbors=args.neighbors)
  # fitting the classifier  
    neigh.fit(X_proj_m, labels_m)
  
    print("kNN classification results: \n")
   
  # the function predict returns the predicted labels for the input data
  # below the predicted labels for training, devel and testing data respectively
    predict_train=neigh.predict(X_proj_m)
    cer_pca_train=CER(labels_train, predict_train)
    print("  CER on training data  = %.2f%% " % (100*cer_pca_train))
    predict_devel=neigh.predict(devel)
    cer_pca_devel=CER(labels_devel, predict_devel)
    print("  CER on devel data  = %.2f%% " % (100*cer_pca_devel))
    predict_test=neigh.predict(test_d)
    cer_pca_test = CER(labels_test, predict_test)
    print("  CER on testing data  = %.2f%% " % (100*cer_pca_test))


  if args.GMM:
  ####### classification with GMM   ###########
    print('Implementing GMM classification')
  # implementing GMM classification using sklearn module
    from sklearn.mixture import GMM
    # number of classes
    nb_classes=numpy.unique(labels_train).shape[0]
  # creating the classifier
    classifier =  GMM(n_components=100, covariance_type=args.cov_type, n_iter=args.n_iter)

  #####   Classification results   ######
  # fitting the classifier
    classifier.fit(X_proj_m, labels_m)

  # the function predict returns the predicted labels for the input data
  # below the predicted labels for training, devel and testing data respectively
    print("GMM classification results: \n")
    predict_train=classifier.predict(X_proj_m)
    cer_lda_train=CER(labels_train, predict_train)
    print("  CER on training data  = %.2f%% " % (100*cer_lda_train))
    predict_devel=classifier.predict(devel)
    cer_lda_devel=CER(labels_devel, predict_devel)
    print("  CER on devel data  = %.2f%% " % (100*cer_lda_devel))
    predict_test=classifier.predict(test_d)
    cer_lda_test = CER(labels_test, predict_test)
    print("  CER on testing data  = %.2f%% " % (100*cer_lda_test))


  if args.nGMM:
  ####### classification with GMM   ###########
    print('Implementing GMM classification2')
  
    nb_classes=numpy.unique(labels_train).shape[0]
  # creating the classifier
    classifier = create_GMM(nb_classes, args.nb_com, args.cov_type, args.n_iter, X_proj_m, labels_m)
    

  #####   Classification results   ######
  # fitting the classifier
    

  # the function predict returns the predicted labels for the input data
  # below the predicted labels for training, devel and testing data respectively
    print("GMM classification results: \n")
    predict_train=GMM_classification(nb_classes, classifier, X_proj_m)
    cer_lda_train=CER(labels_train, predict_train)
    print("  CER on training data  = %.2f%% " % (100*cer_lda_train))
    predict_devel=GMM_classification(nb_classes, classifier, devel)
    cer_lda_devel=CER(labels_devel, predict_devel)
    print("  CER on devel data  = %.2f%% " % (100*cer_lda_devel))
    predict_test=GMM_classification(nb_classes, classifier, test_d)
    cer_lda_test = CER(labels_test, predict_test)
    print("  CER on testing data  = %.2f%% " % (100*cer_lda_test))
  
 
  #print "ok"

  
###########          plotting the confusion matrices           ##################

  # number of confusion matrices
  N = 3
  
  #plotting the figures
  fig = mpl.figure(figsize=(N*6, 6))

  def plot_cm(X, y, set_name):

  # plot training
    cm = data1.confusion_matrix(X, y)
    res = mpl.imshow(cm, cmap=mpl.cm.summer, interpolation='nearest')

    for x in numpy.arange(cm.shape[0]):
      for y in numpy.arange(cm.shape[1]):
        col = 'white'
        if cm[x,y] > 0.5: col = 'black'
        mpl.annotate('%.2f' % (100*cm[x,y],), xy=(y,x), color=col, fontsize=8, horizontalalignment='center', verticalalignment='center')

    classes = [str(k) for k in range(10)]

    mpl.xticks(numpy.arange(10), classes)
    mpl.yticks(numpy.arange(10), classes, rotation=90)
    mpl.ylabel("(Your prediction)")
    mpl.xlabel("(Real class)")
    mpl.title("Confusion Matrix (%s set) - in %%" % set_name)

  mpl.subplot(1, N, 1)
  plot_cm(labels_train, predict_train, 'train')
  mpl.subplot(1, N, 2)
  plot_cm(labels_devel, predict_devel, 'devel.')
  mpl.subplot(1, N, 3)
  plot_cm(labels_test, predict_test, 'test')

  print("Close the plot window to terminate.")
  mpl.show()



  # creates a matrix for the MLP output in which only the correct label
  # position is set with +1.
  #y_train = -1*numpy.ones((10, len(labels_train)), dtype=float)
  #y_train[labels_train, range(len(labels_train))] = 1.0
  #y_devel = -1*numpy.ones((10, len(labels_devel)), dtype=float)
  #y_devel[labels_devel, range(len(labels_devel))] = 1.0

  #print("Using %d samples for training..." % len(y_train.T))

  # Normalizing input set
  #print("Normalizing input data...")
  #X_train  = X_train.astype(float)
  #X_train /= 255.
  #X_mean   = X_train.mean(axis=1).reshape(-1,1)
  #X_std   = X_train.std(axis=1, ddof=1).reshape(-1,1)
  #X_train = (X_train - X_mean) / X_std
  #X_train -= X_mean

  #import project as answers

  #trainer = answers.Trainer(args.seed, args.hidden, args.regularization, args.projected_gradient_norm)
  #start = time.time()
  #machine = trainer.train(X_train, y_train*0.8)
  #total = time.time() - start

  #if machine is None:
  #  print("Training did **NOT** finish. Aborting...")
  #  sys.exit(1)

  #sys.stdout.write("** Training is over, took %.2f minute(s)\n" % (total/60.))
  #sys.stdout.flush()

  #f = bob.io.base.HDF5File(args.machine_file, 'w')
  #f.set('X_mean', X_mean)
  #machine.save(f)
  #del f

  #X_devel = X_devel.astype(float)
  #X_devel /= 255.
  #X_devel -= X_mean

  #print("** Development set results (%d examples):" % X_devel.shape[1])
  #print("  * cost (J) = %f" % machine.J(X_devel, y_devel))
  #cer = machine.CER(X_devel, y_devel)
  #print('  * CER      = %g%% (%d sample(s))' % (100*cer, X_devel.shape[1]*cer))

if __name__ == '__main__':
  main()
