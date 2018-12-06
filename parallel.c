#define N 100
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#define  Max(a,b) ((a)>(b)?(a):(b))

int THREADS = 1;
#define TRIES 7
double   maxeps = 0.1e-7;
int itmax = 100;
int i,j,k;

double eps;
double A [N][N][N];
double times[TRIES];

void relax();
void init();
void verify();

int main(int an, char **as)
{
  THREADS = strtol(as[1], NULL, 10);
  omp_set_num_threads(THREADS);
	int it;
  
  
  double time = omp_get_wtime();
  
	init();
	for(it=1; it<=itmax; it++)
	{
		eps = 0.;
		relax();
		printf( "it=%4i   eps=%f\n", it,eps);
		if (eps < maxeps) break;
	}
	verify();
  
  time = omp_get_wtime() - time;
  
  printf("Threads: %i\nTime: %4.2f seconds\n", THREADS, time);
  
	return 0;
}

void relax()
{

    for(i=1; i<=N-2; i++) {
        #pragma omp parallel for private (k, j) shared(A, i) collapse(2)
        for (j = 1; j <= N - 2; j++) {
            for (k = 1; k <= N - 2; k++) {
                A[i][j][k] = (A[i - 1][j][k] + A[i + 1][j][k]) / 2.;
            }
        }
    }

    #pragma omp parallel for private (k, j, i) shared(A) collapse(2)
    for(i=1; i<=N-2; i++) {
        for (k = 1; k <= N - 2; k++) {
            for (j = 1; j <= N - 2; j++) {
                A[i][j][k] = (A[i][j - 1][k] + A[i][j + 1][k]) / 2.;
            }
        }
    }

    #pragma omp parallel for private (k, j, i) shared(A) reduction(max:eps) collapse(2)
    for(i=1; i<=N-2; i++) {
        for (j = 1; j <= N - 2; j++) {
            for (k = 1; k <= N - 2; k++) {
                double e;
                e = A[i][j][k];
                A[i][j][k] = (A[i][j][k - 1] + A[i][j][k + 1]) / 2.;
                eps = Max(eps, fabs(e - A[i][j][k]));
            }
        }
    }
}

void verify()
{
    double s;

    s=0.;
    #pragma omp parallel for private (k, j, i) shared(A) collapse(3) reduction(+:s)
    for(i=0; i<=N-1; i++)
        for(j=0; j<=N-1; j++)
            for(k=0; k<=N-1; k++)
            {
                s=s+A[i][j][k]*(i+1)*(j+1)*(k+1)/(N*N*N);
            }
    printf("  S = %f\n",s);
}
