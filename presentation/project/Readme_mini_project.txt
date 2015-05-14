mini_project.py loads all the data-sets (training, development and testing) and returns the CER for all of the data-sets. 
In order to run the classification functions, the user should install the module sklearn, following the instructions:

copy-paste the following commands in your terminal

	sudo apt-get install build-essential python-dev python-setuptools \python-numpy python-scipy \libatlas-dev libatlas3gf-base
	sudo apt-get install python-matplotlib
	pip install --user --install-option="--prefix=" -U scikit-learn

(for more information visit the website: http://scikit-learn.org/dev/install.html)

The user could choose to implement dimensionality reduction or not, having as options LDA or PCA. Moreover, the user 
should choose one classification method between k-Nearest Neighbors or GMM. (for more information and documentation about the module
visit the website: http://scikit-learn.org)

mini_project.py has the following input arguments:

'-p', '--plot': 		Turn-ON plotting **after** training (default is set to False)

'-PCA', '--PCA': 		Dimensionality reduction with PCA (default is set to False)

  

'-c', '--components': 		Number of principal components to keep (default is set to 5)



'-LDA', '--LDA': 		Dimensionality reduction with LDA (default is set to False)


'-kNN', '--kNN': 		Classification using kNN method (default is set to Flase)

'-nn', '--neighbors': 		Number of neighbors for kNN classification (default is set to 9")

 

'-GMM', '--GMM': 		Classification using GMM method (default is set to False)



'-n_iter', '--n_iter': 		Number of EM iterations on GMM classification (default is set to 1000000)")

  

'-cov_type', '--cov_type': 	String describing the type of covariance parameters on GMM classification. 
				Must be one of 'spherical', 'tied', 'diag' or 'full' (default is set to full)


'-nb_gaus', '--nb_gaus':	Number of gaussians on GMM classification (default is set to 10)

data1.py contains the functions for PCA, LDA, projection and confusion matrix computation that are used in train1.py

Please contact us for more information about the code and the algorythms.

Sina Mirrazavi 		sina.mirrazavi@epfl.ch
Iason Batzianoulis 	iason.batzianoulis@epfl.ch













