#include "forwardProp.h"

double sigmoidPrime(double x){
	double y = exp(-x)/pow((1 + exp(-x)), 2.0);
	return y;
}

double sigmoid(double x){
	double y = 1 / (1 + exp(-x));
	return y;
}

/* Multiplication of two n*m matrices */
double *multiply(double *x, double *y, int n){
	int i;
	double *result = calloc(n, sizeof(double));
	for(i = 0; i < n; i++){
		result[i] = x[i] * y[i];
	}
	return result;
}

/* x is of n * 1, y is of 1 * k */
double **arrayDot(double *x, double *y, int n, int k){
	int i, j;
	double **result = calloc(n, sizeof(double*));
	for(i = 0; i < n; i++){
		result[i] = calloc(k, sizeof(double));
	}
	for(i = 0; i < n; i++){
		for(j = 0; j < k; j++){
			result[i][j] = x[i] * y[j];
		}
	}
	return result;
}

/* x is of n * m, y is of m * k */
double **dot(double **x, double **y, int n, int m, int k){
	int i, j, a;
	double **result = calloc(n, sizeof(double*));
	for(i = 0; i < n; i++){
		result[i] =calloc(k, sizeof(double));
	}
	for(i = 0; i < n; i++){
		for(j = 0; j < k; j++){

			for(a = 0; a < m; a++){
				result[i][j] += x[i][a] * y[a][j];
			}

		}
	}
	return result;
}
/* x is of m*n, y is of m*k, result should be n*k */
double **transDot(double **x, double **y, int n, int m, int k){
	int i;
	int j;
	int a;
	double **result = calloc(n, sizeof(double*));
	for(i = 0; i < n; i++){
		result[i] = calloc(k, sizeof(double));
	}
	for(i = 0; i < n; i++){
		for(j = 0; j < k; j++){
			for(a = 0; a < m; a++){
				result[i][j] += x[a][i] * y[a][j];
			}
		}
	}
	return result;
}

 /*x is of n*k, y is of n*1 */
double *arrayTranDot(double **x, double *y, int n, int k){
	int i, j;
	double *result = calloc(k, sizeof(double));
	for(i = 0; i < k; i++){
		for(j = 0; j < n; j++){
			result[i] += x[j][i] * y[j];
		}
	}
	return result;
}

void printMatrix(double **x, int n, int m){
	int i, j;
	for(i = 0; i < n; i++){
		for(j = 0; j < m; j ++){
			printf("  %f", x[i][j]);
		}
		printf("\n");
	}
}


// forward propagation

/*  x is of n*m;
	w is of (m+1) * k
*/
double **forward1(double **x, double **w, int n, int m, int k){
	double ** result = dot(x, w, n, m, k);
	return result;
}

/* x is of n*k */
double **sigForward1(double **x, int n, int k){
	int i, j;
	double **result = calloc(n, sizeof(double *));
	for(i = 0; i < n; i++){
		result[i] = calloc(k, sizeof(double));
	}
	for(i = 0; i < n; i++){
		for(j = 0; j < k; j++){
			result[i][j] = sigmoid(x[i][j]);
		}
	}
	return result;
}

/*x is of n*k, y is of k*1 */
double *forward2(double **x, double *y, int n, int k){
	int i, j;
	double *result = calloc(n, sizeof(double));
	for(i = 0; i < n; i++){
		for(j = 0; j < k; j++){
			result[i] += x[i][j] * y[j];
		}
	}
	return result;
}

double* sigForward2(double *x, int n){
	int i;
	double *result = calloc(n, sizeof(double));
	for(i = 0; i < n; i++){
		result[i] = sigmoid(x[i]);
	}
	return result;
}

