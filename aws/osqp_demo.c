#include "osqp.h"
#include <stdlib.h>
#include <sys/time.h>
#include <stdio.h>

typedef struct {
OSQPInt    n;
OSQPInt    m;
OSQPCscMatrix     *P;
OSQPFloat *q;
OSQPCscMatrix     *A;
OSQPFloat *l;
OSQPFloat *u;
} OSQPTestData;

long timeNow(){
    // Special struct defined by sys/time.h
  struct timeval tv;
  // Long int to store the elapsed time
  long fullTime;
  // This only works under GNU C I think
  gettimeofday(&tv, NULL);
  // Do some math to convert struct -> long
  fullTime = tv.tv_sec * 1000000 + tv.tv_usec;
  return fullTime;
}

OSQPFloat *read_float_vec_from_file(FILE *file_ptr, OSQPInt vec_len){
  OSQPFloat * vec_data = (OSQPFloat*) malloc(vec_len * sizeof(OSQPFloat));
  	fread(vec_data,
		sizeof(OSQPFloat), 
		vec_len, 
		file_ptr);
  return vec_data;
}

OSQPInt *read_int_vec_from_file(FILE *file_ptr, OSQPInt vec_len){
  OSQPInt * vec_data = (OSQPInt*) malloc(vec_len * sizeof(OSQPInt));
  fread(vec_data,
		sizeof(OSQPInt), 
		vec_len, 
		file_ptr);
  return vec_data;
}

OSQPCscMatrix *read_OSQPCscMatrix_from_file(FILE *file_ptr,
						OSQPInt m, 
						OSQPInt n, 
						OSQPInt nzmax)
{
  OSQPCscMatrix * A = (OSQPCscMatrix*) malloc(sizeof(OSQPCscMatrix));
  A->m = m;
  A->n = n;
  A->nz = -1;
  A->nzmax = nzmax;
  A->x =  read_float_vec_from_file(file_ptr, A->nzmax);
  A->i =  read_int_vec_from_file(file_ptr, A->nzmax);
  A->p =  read_int_vec_from_file(file_ptr, n+1);
  return A;
}

#define PROBLEM_DATA_INFO_LEN 8
OSQPTestData * read_problem_data(const char * filename)
{
  OSQPTestData * data = (OSQPTestData *)malloc(sizeof(OSQPTestData));
  unsigned int data_info[PROBLEM_DATA_INFO_LEN];
  FILE * file_ptr = fopen(filename, "rb");
  if (file_ptr == NULL) {
	  perror("Failed to open problem data file");
	  return NULL;
  }
  /* read problem dimension */
  fread(data_info,
		sizeof(unsigned int), 
		PROBLEM_DATA_INFO_LEN, 
		file_ptr);
  data -> n = data_info[0];
  data -> m = data_info[1];
  printf("read problem data n = %d, m = %d\n", data->n, data->m);
  // printf("size of OSQPFloat %ld\n", sizeof(OSQPFloat));

  /* read vector l*/
  data->l = read_float_vec_from_file(file_ptr, data->m);
  /* read vector u*/
  data->u = read_float_vec_from_file(file_ptr, data->m);
  /* read vector q*/
  data->q = read_float_vec_from_file(file_ptr, data->n);
  /* read matrix A */
  data->A = read_OSQPCscMatrix_from_file(
    file_ptr,
		data->m,
		data->n,
		data_info[2]);
  /* read matrix P */
  data->P = read_OSQPCscMatrix_from_file(file_ptr,
		data->n,
		data->n,
		data_info[3]);

  fclose(file_ptr);

  return data;
}

int main(int argc, char *argv[])
{
  if(argc!=2){
	printf("provide the input file, argc: %d\n", argc);
	return 0;
  }
  /* read data in run time instead from header file */
  OSQPTestData *data; 
  long time_start, time_end;
  time_start = timeNow();

  data=read_problem_data(argv[1]);

  time_end = timeNow();
  printf("Problem data reading time: %ld us\n", time_end - time_start);

  /* Exitflag */
  OSQPInt exitflag;
  /* Solver, settings, matrices */
  OSQPSolver *solver = NULL;
  OSQPSettings *settings = NULL;

  settings = (OSQPSettings *)malloc(sizeof(OSQPSettings));
  if (settings){
	  osqp_set_default_settings(settings);
  }

  /* Setup solver */
  exitflag = osqp_setup(&solver, 
  	data->P, 
		data->q,
		data->A, 
		data->l, 
		data->u,
		data->m, 
		data->n, 
		settings);

  /* profile continous solve */
  for (int i=0;i<5;i++){
    if (!exitflag) {
      exitflag = osqp_update_data_vec(solver, 
      data->q, 
      data->l, 
      data->u);
    }

    if (!exitflag) {
      exitflag = osqp_update_data_mat(solver,
        data->P->x, 
        OSQP_NULL, 
        data->n,
        data->A->x, 
        OSQP_NULL, 
        data->m);
    }

    if (!exitflag) exitflag = osqp_solve(solver);
  }

  /* Cleanup */
  osqp_cleanup(solver);
  if (settings) free(settings);
  return (int)exitflag;

}