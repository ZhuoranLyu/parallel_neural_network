#ifndef BACKPROP_H_
#define BACKPROP_H_

double costFunction(double* yHat, double* y, int n);

double** costFunctionPrime(double* yHat, double* y, double** z2, double* z3, double** a2, double** W, double** X,int n, int k, int m);

#endif