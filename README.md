### Multiclass logistic regression implementation in Python

An adaptation of week 4 programming exercise from "Machine Learning by Stanford University" in coursera.

Concretely, the goal is to train a linear classifier to predict handrwitten numbers from 0 to 9. This is achieved by using logistic regression and classifying multiple classes using a one-vs-all approach. The training itself its performed against a MNIST subset of 5000 examples contained in the `ex3data1.mat` file.

Jupyter notebook file `multiclass-logistic-regression.ipynb` outlines the general steps of the process, perfmorming data modeling representation and gradient descent optimization on the training set. Scipy and Numpy libraries are used for matrix operations and cost function minimization.
