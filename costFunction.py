import numpy as np
from sigmoid import sigmoid

#returns [J, grad]

def costFunction(theta, X, y, l):
	
	grad = np.zeros(theta.shape)
	
	m = len(X)

	h = sigmoid(X.dot(theta))

	reg_theta = np.concatenate(([[0.]], theta[1:]))
	
	p = (l/(2*m)) * (reg_theta.T.dot(reg_theta))

	J = (1.0/m) * (-y.T.dot(np.log(h)) - (1 - y).T.dot(np.log(1 - h))) + p
	
	grad = 1.0/m * (X.T.dot(h - y)) + l * reg_theta * (1.0/m)

	return J, grad

def optimizableCostFunction(theta, *args):
	X, y, l = args

	theta = np.reshape(theta, (len(theta),1))

	m = len(X)

	h = sigmoid(X.dot(theta))

	reg_theta = np.concatenate(([[0.]], theta[1:]))
	
	p = (l/(2*m)) * (reg_theta.T.dot(reg_theta))

	J = (1.0/m) * (-y.T.dot(np.log(h)) - (1 - y).T.dot(np.log(1 - h))) + p

	return J[0, 0]
	

def gradFunction(theta, *args):
    X, y, l = args

    #reshape theta
    theta = np.reshape(theta, (len(theta),1))

    m = len(X)

    h = sigmoid(X.dot(theta))
 
    reg_theta = np.concatenate(([[0.]], theta[1:]))

    grad = 1.0/m * (X.T.dot(h - y)) + l * reg_theta * (1.0/m)

    return grad.flatten()

