=====================================
 Mini-Project: Pen Digit Recognition - pca+lda
=====================================

These are instructions for your mini-project, which is to create a system for
recognizing hand-written digits using the techniques you have learned in the
classroom so far. You're allowed to use any of the methods described. Your
solution should be written in the Python language using the provided tools, as
this is a simulation of the final project. You should not spend more than a few
hours on this task.

The data you'll have available for this project was collected using a trackpad
and a stylus. It contains the information of the trace each person did for
writing down the numbers. The points are given in drawing order, which can be
used to improve recognition. This is a different aspect when comparing to the
M-NIST Digit Recognition problem we saw during the lab exercises, in which
users only have access to the images of digits and, thus, not the
point-by-point drawing order. You may use this information in your favor or
not, but you should justify why you decided to do so or not.

You must group together in sets of 2 or 3 people for this mini-project.

This mini-project is **not** subject to grading. It is just a simulation of how
the final project may be. If you complete this project well, you shall be
prepared to tackle the final project on your own without major issues.


How to Approach the Problem
---------------------------

Study your database, check the number of examples you have and try from the
simplest to the most complex machine learning tools. You can use anything you
have learned through your lab lessons.

You should try at least 2 techniques. If you followed our labs and completed
the exercise, you should not spend more than 2 hours on it.

Your solution will try to beat my baseline (~50% CER at the test set - easy;-).
As a result, you should provide a confusion matrix (use my example code). The
training, development and testing CERs.

Your solution will be reproducible, so other colleagues should be able to
understand it and **execute** it.

On the next lab, 15th of May, your solution will be presented by another group.
So, this group can do it well, you must provide them:

* A small report that explains what you did and why you did it - this can be
  done orally or in written format
* The code you used to devise the solutions
* Instructions to reproduce your results

Each group will have 5 minutes of presentation time. The objective is to
present what another group did and comment on it (goods and bads).

Reproducing my Baseline
-----------------------

My baseline use the neural networks we have developed during Lab 2. Here is how
to reproduce it::

  $ ./train.py --hidden 3 --regularization 0.001 --projected-gradient-norm 1e-7 mlp.hdf5
  $ ./test.py --test --plot mlp.hdf5

You have slides with details of this method (See Lab 2, exercise 3), as well as
the meaning of the parameters.

The API to the file ``data.py`` contains 2 main functions of interest for your
project:

* ``load()``: The main function to load one of the HDF5 datafiles inside the
  ``data`` subdirectory. This function returns the point-clouds and is
  documented.
* ``as_mnist()``: This function transform the point-clouds into a (binary)
  image, of the same dimensions as on the M-NIST problem which makes it
  convenient to use with our exercise.

You may use any of the two APIs to access the input data, for as long as you
remain consistent through the whole exercise. **Don't mix**. My baseline uses
``as_mnist()``.

Note: You can use ``data.py`` to visualize some of the digits in the database,
with the following command::

  $ python ./data.py

In case of doubt, post a message to the mailing list.

Data Format
-----------

You will not need to "understand" the data format itself, as we are providing
you an API to access the data, but in case you're curious:

There are 4 HDF5 files:

  * train: Use this one for training
  * devel: Use data in this file for development (evaluation)
  * test-dependent: Writer dependent test
  * test-independent: Writer-independent test

Each HDF5 contains 10 folders named after the digit class ("0".."9"). Each
folder contains 2 arrays named "points", 64-bit float with two columns for the
(x, y) coordinates of each point avaiable (these are numbers between [0, 1] and
correspond to the pen position inside the writing pad) and "sizes", which
correspond to the size of each example in the "points" array.

Notice that all points for all examples are packed inside the "points" array.
The reading code must use the "sizes" parameter to effectively select the
points from that array that actually pertain to the example. So, for example,
if "sizes" contains [72, 65], it means that the first example of the digits of
the given class contains 72 points (the 72 first points) and the second example
contains 65 (the following 65 points) - and so on.
