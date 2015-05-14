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

  parser.add_argument('-nb_gaus', '--nb_gaus', default=10, type=int, help="Number of gaussians on GMM classification (defaults to %(default)s)")
  


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
    numpy.random.seed(6)
    # number of classes
    nb_classes=numpy.unique(labels_train).shape[0]
  # adding the labels as an extra feature
    X_proj_m=numpy.matrix(X_proj_m)
    labels_m=numpy.matrix(labels_m)

    X_proj_m_a=numpy.append(X_proj_m,numpy.transpose(labels_m), axis=1)  
  # creating the classifier
    classifier =  GMM(n_components=args.nb_gaus, covariance_type=args.cov_type, n_iter=args.n_iter)

  #####   Classification results   ######
  # fitting the classifier
    classifier.fit(X_proj_m_a)

  # the function predict returns the predicted labels for the input data
  # below the predicted labels for training, devel and testing data respectively
    print("GMM classification results: \n")
  # creating a list for storing the data adding each time one label as an extra dimension. This would help us to calculate the probability 
  # of each sample to be in each class. After calculating the above probability, the predicted label will be the one that has the highest 
  # value of probability
    Xprm={}
    for i in range(nb_classes):
      X_1=numpy.zeros((X_proj_m.shape[0],1))
      X_1.fill(i)
      Xprm[i]=numpy.append(X_proj_m,X_1,axis=1)
  # creating a matrix to store the classification results
    predict_train=numpy.ones((X_proj_m.shape[0],nb_classes))

    for i in range(nb_classes):
  # saving the probability of each sample among the classes
      predict_train[:,i]=numpy.exp(classifier.score(Xprm[i]))
  # finding the higher likelihood between the classes for every sample (higher wins). This represents the predicted label
    #import ipdb; ipdb.set_trace();
    predict_train=numpy.argmax(predict_train,axis=1)
	# calculating the classification error
    cer_train=CER(labels_train, predict_train)
    print("  CER on training data  = %.2f%% " % (100*cer_train))
    # the same for development data and testing data


    Xprm={}
    for i in range(nb_classes):
      X_1=numpy.zeros((devel.shape[0],1))
      X_1.fill(i)
      Xprm[i]=numpy.append(devel,X_1,axis=1)
    predict_devel=numpy.ones((devel.shape[0],nb_classes))
    for i in range(nb_classes):
      predict_devel[:,i]=numpy.exp(classifier.score(Xprm[i]))
    predict_devel=numpy.argmax(predict_devel,axis=1)
    cer_devel=CER(labels_devel, predict_devel)
    print("  CER on devel data  = %.2f%% " % (100*cer_devel))
    

    Xprm={}
    for i in range(nb_classes):
      X_1=numpy.zeros((test_d.shape[0],1))
      X_1.fill(i)
      Xprm[i]=numpy.append(test_d,X_1,axis=1)
    predict_test=numpy.ones((test_d.shape[0],nb_classes))
    for i in range(nb_classes):
      predict_test[:,i]=numpy.exp(classifier.score(Xprm[i]))
    predict_test=numpy.argmax(predict_test,axis=1)
    cer_test = CER(labels_test, predict_test)
    print("  CER on testing data  = %.2f%% " % (100*cer_test))
  
 
  

  
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



 

if __name__ == '__main__':
  main()
