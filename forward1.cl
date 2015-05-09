/*  x is of n*m;
	w is of (m+1) * k
*/
double **forward1(double **x, double **w, int n, int m, int k){
	double ** result = dot(x, w, n, m, k);
	return result;
}

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
double sigmoid(double x){
	double y = 1 / (1 + exp(-x));
	return y;
}
=================================================
kernel void array_dot(
      global float* a,
      global float* b,
      global float* c,
      global float* d,
      const int n,
      const int k,
      const int m)
{
  int x, y, i;
  x = get_global_id(0);
  y = get_global_id(1);
  float val = 0.0;
  for(i = 0; i < m; i++){
    val += a[x * m + i] * b[i * k + y];
  }
  c[x * k + y] = val;
  d[x * k + y] = 1 / (1 + exp(-val));
  return;
}