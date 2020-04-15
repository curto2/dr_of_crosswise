#   Doctor of Crosswise: Reducing Over-parametrization in Neural Networks.		    
#										
#   Authors: Curtó and Zarza.
#   c@decurto.tw z@dezarza.tw 						    

import numpy as c
import time as t
from sklearn import metrics
from sklearn.datasets import fetch_openml as ml
from sklearn.ensemble import RandomForestClassifier

ratio_of_features = 2;

def selu(C):
	C = c.multiply(1.0507009873554804934193349852946, C)
	C[C <= 0] = c.multiply(1.6732632423543772848170429916717, c.expm1(C[C <= 0]))
	return C

def c_and_z(A, B):
	M = A.shape[0] 
	N = B.shape[0] 
	C = c.zeros((M, A.shape[1] * N))
	Z = c.zeros((N, A.shape[1]))
	for m in range(M):
		for n in range(N):
			Z[n, :] = c.multiply(A[m, :], B[n, :])
		C[m, :] = Z.flatten()
	return C

def run():
	mnist = ml('mnist_784')
	n_train = 60000
	n_test = 10000

	c.random.seed(7)

	# Define sets of train and test.
	train = c.arange(0, n_train)
	test = c.arange(n_train + 1, n_train + n_test)

	X_train, Y_train = mnist.data[train], mnist.target[train]
	X_test, Y_test = mnist.data[test], mnist.target[test]

	# Apply an algorithm of learning.
	print("Applying an algorithm of learning.")
	clr = RandomForestClassifier(n_estimators = 100, n_jobs = -1)

	K = c.random.randn(ratio_of_features, X_train.shape[1])
	b = c.random.randn(X_train.shape[1] * ratio_of_features)

	x1 = selu(c_and_z(X_train, K) + b)
	K1 = c.random.randn(2, x1.shape[1])
	b1 = c.random.randn(2 * x1.shape[1])

	x2 = selu(c_and_z(x1, K1) + b1)
	K2 = c.random.randn(x2.shape[1])
	b2 = c.random.randn(x2.shape[1])

	clr.fit(selu(c.multiply(x2, K2) + b2), Y_train)

	# Make a prediction.
	print("Making predictions.")

	X_t1 = selu(c_and_z(X_test, K) + b)
	X_t2 = selu(c_and_z(X_t1, K1) + b1)

	Y_p = clr.predict(selu(c.multiply(X_t2, K2) + b2))

	# Evaluate the prediction.
	print("Evaluating results...")
	print("Precision: \t \t", "{:.2f}".format(metrics.accuracy_score(Y_test, Y_p) * 100), "%.")

if __name__ == "__main__":
	start = t.time()
	run()
	end = t.time()
	print("Overall running time: \t", "{:.2f}".format(end - start), "s.")
