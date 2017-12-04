## Multiclass logistic regression implementation in Python

Octave/Python adaptation of week 4 programming exercise from "Machine Learning by Stanford University" course in coursera.

Concretely, the goal is to train a linear classifier to predict handrwitten numbers from 0 to 9. This is achieved by using logistic regression and classifying multiple classes using a one-vs-all approach. The training itself its performed against a MNIST subset of 5000 examples contained in the `ex3data1.mat` file.

Jupyter notebook file `multiclass-logistic-regression.ipynb` outlines the general steps of the process, perfmorming data modeling representation and gradient descent optimization on the training set. Scipy and Numpy libraries are used for matrix operations and cost function minimization.

This implementation uses numpy ndarrays to represent the training set and matrix operations, scipy for optimize the cost function using minimize BFGS method, and importing matlab dataset. `costFunction.py` offers a pure implementation of a cost function for linear regression along with gradient descent using regularization on both cases, and separated, "scipy-optimizable" cost and gradient functions. 

Algorithm with given notebook settings achieves a ~96% accuracy.

### Caveats on the training set:

MNIST training set was externally packaged in matlab format, vars used consist on a 5000x400 'X' matrix, used for the training examples, and a 5000x1 'y' matrix representing its respective categories. Each of the rows represents a 20x20 pixels image, thus the 400 columns. Columns on the X matrix, represents a pixel intensity in grayscale from 0 to 1, these are the features on our training set. On the other hand, the 'y' matrix holds the 10 categories, this is, numbers from 0 to 9, where 0 is mapped to 10 for practical purposes on the original programming exercise goal for OCTAVE/Matlab non-zero indexed structures. 

### Installation

For simplicity, it is recommended to install Conda, download content of this repository and run the jupyter notebook as it is.

### Further work

This code can be modularized and generalized to create a fully adaptable classifier for any number of K classes, and X, y training sets like https://github.com/scikit-learn-contrib/lightning, etc. Although this implementation´s main objective was merely for academic purposes.
