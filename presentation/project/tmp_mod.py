def lda(X, y):
  """Calculates the projection matrix U to perform LDA on X with labels y.

  LDA finds the projecting matrix W that allows us to linearly project X to
  another (sub) space in which the between-class and within-class variances are
  jointly optimized: the between-class variance is maximized while the
  with-class is minimized. The (inverse) cost function for this criteria can be
  posed as the following:

    J(W) = W^T Sb W / W^T Sw W

  Where:

    W - the transformation matrix that converts X into the LD space
    Sb - the between-class scatter; it has dimensions (X.shape[0], X.shape[0])
    Sw - the within-class; it also has dimensions (X.shape[0], X.shape[0])

  N.B.: A scatter matrix equals the covariance matrix if we remove the division
  factor.

  Because this cost function is convex, you can just find its maximum by
  solving dJ/dW = 0. As exposed in class, this problem can be re-formulated as
  finding the eigen values (l_i) that solve the following condition:

    Sb = l_i Sw, or

    (Sb - l_i Sw) = 0

  If you can do that, then the respective eigen vectors that correspond to the
  eigen values l_i form W.

  To calculate Sb and Sw, you start by computing the class means, for each
  class c:

    mu_c = mean(X when y == c)

  Then the overall mean:

    mu = mean(n_c * mu_c)

  Sb is easy, it is the scatter matrix considering mu_c and mu

    Sb = sum_C ( n_c (mu_c - mu)(mu_c - mu)^T )

  Sw looks the same, but we compute the scatter over each individual component
  for each class and sum all up:

    Sw = sum_C ( sum_n_c (x_i^c - mu_c) * (x_i^c - mu_c)^T )

  You can call ``numpy.linalg.eig`` to solve the problem (Sb - l_i Sw).

  Keyword arguments:

  X
    The data matrix in which each column corresponds to one observation of the
    data and each row, to one feature.

  y
    The label vector containing the labels of each column of X

  TODO: You must return e, the eigen values and U, the eigen vectors of the
  generalized Eigen decomposition for LDA, computed from the data. U must be
  arranged such that each column represents one eigen vector. For example, to
  project X on U we would do: X' = U^T X.
  """
  
  # Replace e and U by your calculations. Read docs above.
  # The shapes of e and U should be similar to the shapes below
  #e = numpy.ones((X.shape[0],), dtype=float)
  #U = numpy.ones((X.shape[0], X.shape[0]), dtype=float)

  # Step 1: compute the class means mu_c, starting from the sum_c
  # Tip: Place the results in the matrix mu_c in which each column represents
  #      the mean for one class.

  mu_c=numpy.array([numpy.mean(X[:,y==k], axis=1)for k in range(10)]).T

  # Step 1.5: computes the number of elements in each class

  n_c=numpy.array([sum(y==k)for k in range (10)])

  # Step 2: computes the global mean mu
  
  
  mu=numpy.sum(mu_c*n_c, axis=1)/X.shape[1]

  # Step 3: compute the between-class scatter Sb
  mu_c_mu=(mu_c.T-mu).T
  Sb=numpy.dot(n_c*mu_c_mu, mu_c_mu.T)

  # Step 4: compute the within-class scatter Sw
  
  Sw=numpy.zeros((X.shape[0], X.shape[0]), dtype=float)
  for k in range(10):
    X_c_mu_c=(X[:,y==k].T-mu_c[:,k]).T
    Sw+=numpy.dot(X_c_mu_c, X_c_mu_c.T)

  # Step 5: calculate the eigen vectors and values of Sw^-1 Sb
  
 

  e, U = numpy.linalg.eig(numpy.dot(numpy.linalg.pinv(Sw),Sb))
  

  return e, U
