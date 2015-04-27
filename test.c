#include <stdlib.h>
#include <stdio.h>
#include <math.h>

int main(int argc, char **argv){

	double** X; // input matrix, m by n
  double* y; // output array, m by 1

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
  int i,j;
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
	int n = lineLen; // example size
	int m = lineNum;

  printf("m = %d, n = %d.\n", m, n);
  for (i = 0; i < n -1; i++){
    printf("%f, ", X[7493][i]);
  }
  printf("\n");
  printf("%f, ", y[7493]);
}