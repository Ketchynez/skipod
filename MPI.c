#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#define  Max(a,b) ((a)>(b)?(a):(b))
#define  Min(a,b) ((a)<(b)?(a):(b))


#ifndef N
#define N 100
#endif
double   maxeps = 0.1e-7;
int itmax = 100;
int i,j,k;
double timer;

double eps;
double A [N][N][N];
int nProcs, rank, step, start_index, end_index;

void relax();
void init();
void verify();

int main(int *argc, char **argv)
{
    int it;

    MPI_INIT(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    step = (N - 2 + nProcs - 1) / nProcs;
    start_index = step * rank + 1;
    end_index = Min(N - 2, start_index + step - 1);

    if (!rank) {
        init();
    }

    MPI_Bcast(A, N * N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    timer = MPI_Wtime();
    for(it=1; it<=itmax; it++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        eps = 0.;
        relax();
        if (!rank) {
    //        printf("it=%4i   eps=%f\n", it, eps);
            if (eps < maxeps) break;
        }
    }
    timer = MPI_Wtime() - timer;
    if (!rank) {
        FILE* f;
        f = fopen("res", "a+");
        fprintf(f, "%d %d %.6f\n", N, nProcs, timer);
        fclose(f);
        verify();
    }

    MPI_Finalize();
    return 0;
}


void init()
{
    for(i=0; i<=N-1; i++) {
        for (j = 0; j <= N - 1; j++) {
            for (k = 0; k <= N - 1; k++) {
                if (i == 0 || i == N - 1 || j == 0 || j == N - 1 || k == 0 || k == N - 1)
                    A[i][j][k] = 0.;
                else A[i][j][k] = (4. + i + j + k);
            }
        }
    }
}

void relax()
{
    for(i=1; i<=N - 2; i++) {
        for (j = start_index; j <= end_index; j++) {
            for (k = 1; k <= N - 2; k++) {
                A[i][j][k] = (A[i - 1][j][k] + A[i + 1][j][k]) / 2.;
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    for(int proc = 0; proc < nProcs; ++proc) {
        int proc_start_index = step * proc + 1;
        int proc_end_index = Min(N - 2, proc_start_index + step - 1);
        int size = proc_end_index - proc_start_index + 1;
        for(i = 1; i <= N - 2; ++i) {
            if (size > 0) {
                MPI_Bcast(&A[i][proc_start_index][0], size * N, MPI_DOUBLE, proc, MPI_COMM_WORLD);
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    for(i = start_index; i <= end_index; i++) {
        for (j = 1; j <= N - 2; j++) {
            for (k = 1; k <= N - 2; k++) {
                A[i][j][k] = (A[i][j - 1][k] + A[i][j + 1][k]) / 2.;
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    double local_eps = 0.0;
    for(i=start_index; i<=end_index; i++) {
        for (j = 1; j <= N - 2; j++) {
            for (k = 1; k <= N - 2; k++) {
                double e;
                e = A[i][j][k];
                A[i][j][k] = (A[i][j][k - 1] + A[i][j][k + 1]) / 2.;
                local_eps = Max(local_eps, fabs(e - A[i][j][k]));
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    for (int proc = 0; proc < nProcs; ++proc) {
        int proc_start_index = step * proc + 1;
        int proc_end_index = Min(N - 2, proc_start_index + step - 1);
        int size = proc_end_index - proc_start_index + 1;
        if (size > 0) {
            MPI_Bcast(&A[proc_start_index][0][0], size * N * N, MPI_DOUBLE, proc, MPI_COMM_WORLD);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Reduce(&local_eps, &eps, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
}

void verify()
{
    double s;
    printf("verify\n");

    s=0.;
    for(k=0; k<=N-1; k++)
        for(j=0; j<=N-1; j++)
            for(i=0; i<=N-1; i++)
            {
                s=s+A[i][j][k]*(i+1)*(j+1)*(k+1)/(N*N*N);
            }
    printf("  S = %f\n",s);
}
