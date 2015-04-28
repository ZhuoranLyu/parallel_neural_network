#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "forwardProp.h"
#include "backProp.h"
//#include "dataReader.h"

int main(int argc, char **argv){

	int n; // example size
	int m; // input layer size
	int k = 12; // hidden layer size

	int i,j,p;

	double** X; // input matrix, n by m
	double* y; // output, n by 1
	
	srand(time(NULL));
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
	X = calloc(lineNum, sizeof(double*));
	y = calloc(lineNum, sizeof(double));

	for (i = 0;i<lineNum;i++) {
		X[i] = calloc((lineLen-1), sizeof(double));
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
	m = lineLen - 1;

	// Normalize the data
	for (i = 0; i < n; i++){
		for (j = 0; j < m; j++){
			X[i][j] /= 100.;
		}
		y[i] /= 10.;
	}

	double J = 10.; // cost
	double** W; // weight matrix, m+1 by k
	double** z2; // n by k
	double** a2; // n by k
	double* z3; // n by 1
	double* yHat; // estimate output, n by 1

	double** dJdW; // combined dJdW1 and dJdW2, m+1 by k

	double threshold = 1;
	double step = 1;

	double** deltaW;
	double** b1;
	double* b2;

	// Init bias and weights for the network
	W = calloc(m+1, sizeof(double*)); // m+1 by k
	
	deltaW = calloc(m+1, sizeof(double*)); // m+1 by k

	for (i = 0; i < m+1; i++){
		W[i] = calloc(k, sizeof(double));
		deltaW[i] = calloc(k, sizeof(double));	
	}

	for (i = 0; i< m+1; i++){
		for (j = 0; j < k; j++){
			W[i][j] = (double)rand() / RAND_MAX;
			deltaW[i][j] = 0.;
		}
	}

	//while(J > threshold){
	for (p = 0; p < 2000; p++){
		z2 = forward1(X, W, n, m, k); // n by k

		a2 = sigForward1(z2, n, k); // n by k

		z3 = forward2(a2, W[m], n, k); // n by 1

		yHat = sigForward2(z3, n); // n by 1
		
		J = costFunction(yHat, y, n);

		dJdW = costFunctionPrime(yHat, y, z2, z3, a2, W, X, n, k, m);

		for (i = 0; i < n; i++){
			free(z2[i]);
		}
		free(z2);
		for (i = 0; i < n; i++){
			free(a2[i]);
		}
		free(a2);
		free(z3);
		free(yHat);

		for (i = 0; i < m+1; i++){
			for (j = 0; j < k; j++){
				deltaW[i][j] = - 0.001 * dJdW[i][j] + 0.1 * deltaW[i][j];
				W[i][j] += deltaW[i][j];
			}
		}

		for (i = 0; i < m+1; i++){
			free(dJdW[i]);
		}
		free(dJdW);

		if (p % 10 == 0)
			printf("%.15f\n", J);
	}
	for (i = 0; i < m+1; i++){
		for (j = 0; j < k; j++){
			printf("%f, ", W[i][j]);
		}
		printf("\n");
	}
}
