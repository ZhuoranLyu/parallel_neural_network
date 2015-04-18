#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// n, m is the dimension of input X, m, k+1 is the dimension of W, which
// consists of 2 weights matrix. The first is m by k, and the second is
// k by 1. 

// W initialized with random number
double** W; // m+1 by k
double** z2 = forward1(); // n by k
double** a2 = sigmoidForward1(); // n by k
double* z3 = forward2(); // n by 1
double* yHat = sigmoidForward2(); // n by 1
double** X; // n by m

double costFunction(double* yHat, double* y, int n){
	double J = 0;
	int i = 0;
	for (i = 0; i < n; i++){
		J += (y[i]- yHat[i]) * (y[i]- yHat[i]);
	}
	return 0.5*J;
}

double** costFunctionPrime(double* yHat, double* y, int n, int k, int m){
	int i, j;
	double* delta3; // n by 1
	double* dJdW2; // k by 1
	double** delta2; // n by k
	double** dJdW1; // m by k
	double** dJdW = calloc(m+1, sizeof(double*)); // m+1 by k

	for (i = 0; i < n; i++){
		y[i] = - (y[i] - yHat[i]);
		z3[i] = sigmoidPrime(z3[i]);
	}
	delta3 = multiply(y, z3, n, 1);

	dJdW2 = transdot(a2, delta3, k, n, 1);
	
	delta2 = dot(delta3, W[m], n, 1, k);

	for (i = 0; i < n; i++){
		for (j = 0; j < k; j++){
			delta2[i][j] *= sigmoidPrime(delta2[i][j]);
		}
	}

	dJdW1 = transdot(X, delta2, m, n, k);
	for (i = 0; i < m; i++){
		dJdW[i] = dJdW1[i];
	}
	dJdW[m] = dJdW2;
	return dJdW;
}