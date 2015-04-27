#include <stdlib.h>
#include <stdio.h>
#include <math.h>

// helper functions

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


// back propagation

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


int main(int argc, char **argv){

	int n; // example size
	int m; // input layer size
	int k = 15; // hidden layer size

	int i,j,p;

	double** X; // input matrix, n by m
	double* y; // output, n by 1
	
	 //read the matrix from file
	FILE *fp;
	char *filename = "pendigits.tra";
	fp = fopen(filename,"r");
	if (fp == NULL) {
		printf("ERROR: unable to read file.\n");
		return -1;
	}
	char* line = NULL;
	size_t len = 0; //line length
	int lineLen = 0; //matrix length
	int lineNum = 0; //matrix height
	int passed = 0;

	//two passes, first pass to determine number of lines and line length
	// second pass to determine line length

	while (getline(&line,&len,fp) != -1) {
		if (passed == 0) {
			char* elts = strtok(line," ,\t");
			while (elts != NULL) {
				lineLen++;
				elts = strtok(NULL," ,\t");
			}
			passed = 1;
			free(elts);
		}
		lineNum++;
	}
	fclose(fp);

	//open again for pass 2
	fp = fopen(filename,"r");
	X = malloc(sizeof(double)*(lineLen-1)*lineNum);
	y = malloc(sizeof(double)*lineNum);

	for (i = 0;i<lineNum;i++) {
		X[i] = malloc(sizeof(double)*lineLen);
	}
	for (i = 0;i<lineNum;i++) {
		getline(&line,&len,fp);
		char* elts = strtok(line," ,\t");
		for (j=0;j<lineLen-1;j++) {
			X[i][j] = strtod(elts,NULL);
			elts = strtok(NULL," ,\t");
		}
		y[i] = strtod(elts,NULL);
		elts = strtok(NULL," ,\t");
		free(elts);
	}
	fclose(fp);

	n = lineNum; // example size
	m = lineLen;

	double J = 10.; // cost
	double** W; // weight matrix, m+1 by k
	double** z2; // n by k
	double** a2; // n by k
	double* z3; // n by 1
	double* yHat; // estimate output, n by 1

	double** dJdW; // combined dJdW1 and dJdW2, m+1 by k
	// below should be in the while loop

	double threshold = 1;
	double step = 5;


/*
		X = calloc(n, sizeof(double*));
		for (i = 0; i < n; i++){
			X[i] = calloc(m, sizeof(double));
		}
		// init

		X[0][0] = 3.;
		X[0][1] = 5.;
		X[1][0] = 5.;
		X[1][1] = 1.;
		X[2][0] = 10.;
		X[2][1] = 2.;

		y = calloc(n, sizeof(double));
		y[0] = 75.;
		y[1] = 82.;
		y[2] = 93.;

		// normalize
		for (i = 0; i < 3; i++){
			for (j = 0; j < 2; j++){
				X[i][j] /= 10.;
			}
			y[i] /= 100.;
		}
*/

	W = calloc(m+1, sizeof(double*)); // m+1 by k
	for (i = 0; i < m+1; i++){
		W[i] = calloc(k, sizeof(double));
	}
	for (i = 0; i< m+1; i++){
		for (j = 0; j < k; j++){
			W[i][j] = 1.;
		}
	}

	//while(J > threshold){
	for (p = 0; p < 100000; p++){
		z2 = forward1(X, W, n, m, k); // n by k
		a2 = sigForward1(z2, n, k); // n by k
		z3 = forward2(a2, W[m], n, k); // n by 1
		yHat = sigForward2(z3, n); // n by 1
		//printf("%f, %f, %f\n", yHat[0], yHat[1], yHat[2]);        
		//printf("%f, %f, %f\n", y[0], y[1], y[2]);     
		
		J = costFunction(yHat, y, n);
		//printf("%f\n", J);

		dJdW = costFunctionPrime(yHat, y, z2, z3, a2, W, X, n, k, m);
		for (i = 0; i < m+1; i++){
			for (j = 0; j < k; j++){
						//printf("%.10f, ", dJdW[i][j]);
				W[i][j] -=  step * dJdW[i][j];
			}
				//printf("\n");
		}
		//printf("\n");
		/*
		for (i = 0; i < m+1; i++){
			for (j = 0; j < k; j++){
						//printf("%.10f, ", W[i][j]);
						//W[i][j] -=  dJdW[i][j];
			}
				//printf("\n");
		}
		*/
		//printf("\n");
		printf("%.10f\n", J);
	}
//}
}
