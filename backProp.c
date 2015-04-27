#include "forwardProp.h"
#include "backProp.h"

double costFunction(double* yHat, double* y, int n){
	double J = 0;
	int i = 0;
	for (i = 0; i < n; i++){
		J += (y[i]- yHat[i]) * (y[i]- yHat[i]);
	}
	return 0.5*J;
}

double** costFunctionPrime(double* yHat, double* y, double** z2, double* z3, double** a2, double** W, double** X,int n, int k, int m){
	int i, j;
	double* delta3; // n by 1
	double* dJdW2; // k by 1
	double** delta2; // n by k
	double** dJdW1; // m by k
	double** dJdW = calloc(m+1, sizeof(double*)); // m+1 by k
	double* temp = calloc(n, sizeof(double)); // store temp value for y

	for (i = 0; i < n; i++){
		temp[i] = - (y[i] - yHat[i]);
		z3[i] = sigmoidPrime(z3[i]);
	}
	delta3 = multiply(temp, z3, n);

	dJdW2 = arrayTranDot(a2, delta3, n, k);
	
	delta2 = arrayDot(delta3, W[m], n, k);

	for (i = 0; i < n; i++){
		for (j = 0; j < k; j++){
			delta2[i][j] *= sigmoidPrime(z2[i][j]);
		}
	}

	dJdW1 = transDot(X, delta2, m, n, k);
	for (i = 0; i < m; i++){
		dJdW[i] = dJdW1[i];
	}
	dJdW[m] = dJdW2;
	return dJdW;
}

