## Multiclass logistic regression implementation in Python

An adaptation of week 4 programming exercise from "Machine Learning by Stanford University" course in coursera.

Concretely, the goal is to train a linear classifier to predict handrwitten numbers from 0 to 9. This is achieved by using logistic regression and classifying multiple classes using a one-vs-all approach. The training itself its performed against a MNIST subset of 5000 examples contained in the `ex3data1.mat` file.

Jupyter notebook file `multiclass-logistic-regression.ipynb` outlines the general steps of the process, perfmorming data modeling representation and gradient descent optimization on the training set. Scipy and Numpy libraries are used for matrix operations and cost function minimization.

#### Caveats on the training set:

MNIST training set was externally packaged in matlab format, vars used consist on a 5000x400 'X' matrix, used for the training examples, and a 5000x1 'y' matrix representing its respective predictions. Each of the rows represents a 20x20 image, thus the 400 columns. Columns on the X matrix, represents a pixel intensity in grayscale from 0 to 1, these are the features on our training set. On the other hand, the 'y' matrix holds the 10 categories, this is, numbers from 0 to 9, where 0 is mapped to 10 for practical purposes on the original programming exercise goal for OCTAVE/Matlab non-zero indexed structures. 
