#include <stdlib.h>
#include <stdio.h>
#include <math.h>

int main(int argc, char **argv){

	double** X; // input matrix, n by m

        //read the matrix from file
        FILE *fp;
        fp = fopen(filename,"r");
        if (fp == NULL) {
          printf("ERROR: unable to read file.\n");
          return -1;
        }
        char* line = NULL;
        size_t len = 0; //line length
        int lineLen = 0; //matrix length
        int linenum = 0; //matrix height

        //two passes, first pass to determine number of lines and line length
        // second pass to determine line length

        int passed = 0;
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
          linenum++;
        }
        fclose(fp);

        //open again for pass 2
        fp = fopen(filename,"r");
        X = malloc(sizeof(double)*lineLen*linenum);
        int i,j;
        for (i = 0;i<linenum;i++) X[i] = malloc(sizeof(double)*lineLen);
        for (i = 0;i<linenum;i++) {
          getline(&line,&len,fp);
          char* elts = strtok(line," ,\t");
          for (j=0;j<lineLen;j++) {
            X[i][j] = strtod(elts,NULL);
            elts = strtok(NULL," ,\t");
          }
          free(elts);
        }
        fclose(fp);
	int n = lineLen; // example size
	int m = linenum;

	int k; // hidden layer size

	double* y // output, n by 1

	double J; // cost
	double** z2; // n by k
	double** a2; // n by k
	double* z3; // n by 1
	double* yHat; // estimate output, n by 1

	double** W; // m+1 by k, initial with random number
	double** dJdW; // combined dJdW1 and dJdW2, m+1 by k
	// below should be in the while loop

	double threshold = 0.1;
	double step = 0.5;

	W = calloc(m+1, sizeof(double*)); // m+1 by k
	for (i = 0; i < m+1; i++){
		W[i] = calloc(k, sizeof(double));
	}
	for (i = 0; i< m+1; i++){
		for (j = 0; j < k; j++){
			W[i][j] = 1.;
		}
	}

	while(J < threshold){
		J = costFunction(yHat, y, n);

		z2 = forward1(X, W, n, m, k); // n by k
		a2 = sigForward1(z2, n, k); // n by k
		z3 = forward2(a2, W[m], n, k); // n by 1
		yHat = sigForward2(z3, n); // n by 1

		dJdW = costFunctionPrime(yHat, y, z2, z3, W, X, n, k, m);
		for (i = 0; i < m+1; i++){
			for (j = 0; j < k; j++){
				W[i][j] -= step * dJdW[i][j];
			}
		}
	}

}



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

	for (i = 0; i < n; i++){
		y[i] = - (y[i] - yHat[i]);
		z3[i] = sigmoidPrime(z3[i]);
	}
	delta3 = multiply(y, z3, n);

	dJdW2 = transdot(a2, delta3, k, n, 1);
	
	delta2 = dot(delta3, W[m], n, 1, k);

	for (i = 0; i < n; i++){
		for (j = 0; j < k; j++){
			delta2[i][j] *= sigmoidPrime(z2[i][j]);
		}
	}

	dJdW1 = transdot(X, delta2, m, n, k);
	for (i = 0; i < m; i++){
		dJdW[i] = dJdW1[i];
	}
	dJdW[m] = dJdW2;
	return dJdW;
}
