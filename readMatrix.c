#include <stdio.h>
#include <stdlib.h>

//returns number of rows in matrix.
int readMatrix(char* filename,double **X){
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

  while (getline(&line,&len,fp) != -1) {
    char* elts = strtok(line," ,\t");
    while (elts != NULL) {
      lineLen++;
      strtok(NULL," ,\t");
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
    for (j=0;j<lenLen;j++) {
      X[i][j] = strtod(elts,NULL);
      strtok(NULL," ,\t");
    }
  }
  fclose(fp);

  return linenum;
}
