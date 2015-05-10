/* a, b are arrays, c is the result
   mat_mult only uses the first dimensional work items
*/
kernel void multiply(
      global float* a,
      global float* b,
      global float* c)
{
  // Determine the amount of padding for this filter
  int i;
  i = get_global_id(0);
  c[i] = a[i] * b[i];
  return;
}

/* a is n*1, y is 1*k, c should be n*k */
kernel void array_dot(
      global float* a,
      global float* b,
      global float* c,
      const int n,
      const int k)
{
  int i, j;
  i = get_global_id(0);
  j = get_global_id(1);
  if(i < n && j < k){
    c[i * k + j] = a[i] * b[j];
  }
  return;
}

/*a is m*n, b is m*k, c is n*k */
kernel void trans_dot(
      global float* a,
      global float* b,
      global float* c,
      const int n,
      const int m,
      const int k)
{
  int x, y, i;
  x = get_global_id(0);
  y = get_global_id(1);
  float val = 0.0;
  for(i = 0; i < m; i++){
    val += a[i * n + x] * b[i * k + y];
  }
  c[x * k + y] = val;
  return;
}

/*a is n*k, b is n*1, c is k*1 */
kernel void array_trans_dot(
      global float* a,
      global float* b,
      global float* c,
      const int n,
      const int k)
{
  int i, j;
  i = get_global_id(0);
  float val = 0.0;
  for(j = 0; j < n; j++){
    val += a[j * k + i] * b[j];
  }
  c[i] = val;
  return;
}

/*a is n*m, b is m*k, c is n*k, d is n*k */
kernel void forward1(
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

/* a is n*k, y is k*1, c is n*1, d is n*1 */
kernel void forward2(
      global float* a,
      global float* b,
      global float* c,
      global float* d,
      const int n,
      const int k)
{
  int i, j;
  i = get_global_id(0);
  float val = 0.0;
  for(j = 0; j < k; j++){
    val += a[i * k + j] * b[j];
  }
  c[i] = val;
  d[i] = 1 / (1 + exp(-val));
  return;
}

/*a is yHat, b is y, c is z3, d is result delta3 */
kernel void cost_mult(
      global float* a,
      global float* b,
      global float* c,
      global float* d,
      const int n,
      const int k)
{
  int i;
  i = get_global_id(0);
  float temp;
  temp = a[i] - b[i];
  c[i] = exp(-c[i])/pow((1 + exp(-c[i])), 2.0);
  d[i] = temp * c[i];
  return;
}


/* a is a2(n*k), b is delta3(n*1), c is W[m](1*k), d is z2(n*k), e is delta2(n*k), f is dJdW2(k*1)
   I assume there are n*k work items
 */
kernel void cost_dot(
      global float* a,
      global float* b,
      global float* c,
      global float* d,
      global float* e,
      global float* f,
      const int n,
      const int k)
{
  int i, j, z;
  i = get_global_id(0);
  j = get_global_id(1);

  /*first dimension would do array transdot*/
  if(i == 0){
    float val = 0.0;
    for(z = 0; z < n; z++){
      val += a[z * k + j] * b[z];
    }
    f[j] = val;
  }
  if(i < n && j < k){
    e[i * k + j] = b[i] * c[j];
  }
  float temp = d[i * k + j];
  e[i * k + j] = e[i * k + j] * exp(-temp)/pow((1 + exp(-temp)), 2.0)
  return;
}
