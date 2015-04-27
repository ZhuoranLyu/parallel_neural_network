#ifndef FOWRARDPROP_H_
#define FOWRARDPROP_H_

double sigmoidPrime(double x);

double sigmoid(double x);

double *multiply(double *x, double *y, int n);

double **arrayDot(double *x, double *y, int n, int k);

double **dot(double **x, double **y, int n, int m, int k);

double **transDot(double **x, double **y, int n, int m, int k);

double *arrayTranDot(double **x, double *y, int n, int k);

void printMatrix(double **x, int n, int m);

double **forward1(double **x, double **w, int n, int m, int k);

double **sigForward1(double **x, int n, int k);

double *forward2(double **x, double *y, int n, int k);

double* sigForward2(double *x, int n);

#endif 