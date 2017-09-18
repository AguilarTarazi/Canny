#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define SIZE 4

int main(int argc, char *argv[]) {
    int numtasks, rank, sendcount, recvcount, source;
    float sendbuf[SIZE*SIZE] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,9.0, 10.0, 11.0, 12.0,13.0, 14.0, 15.0, 16.0};
    float recvbuf[SIZE];
    // float sendbuf[SIZE];
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    if (numtasks == SIZE) {
        // define source task and elements to send/receive, then perform collective scatter
        source = 1;
        sendcount = SIZE;
        recvcount = SIZE;
        MPI_Scatter(sendbuf,sendcount,MPI_FLOAT,recvbuf,recvcount,
            MPI_FLOAT,0,MPI_COMM_WORLD);
        printf("rank= %d Results: %.1f %.1f %.1f %.1f\n",rank,recvbuf[0],recvbuf[1],recvbuf[2],recvbuf[3]);
        sleep(5);
        MPI_Gather (recvbuf,sendcount,MPI_FLOAT,sendbuf,recvcount,MPI_FLOAT,0,MPI_COMM_WORLD);
        if(rank==0){
            for (int i = 0; i < SIZE*SIZE; i++) {
                printf("%.1f - ", sendbuf[i]);
            }
        }
        printf("\n");
    }
    else
    printf("Must specify %d processors. Terminating.\n",SIZE);
    MPI_Finalize();
}


// #include <mpi.h>
// #include <stdio.h>
// #include <stdlib.h>
//
// int main(int argc, char **argv) {
//     int size, rank;
//
//     MPI_Init(&argc, &argv);
//     MPI_Comm_size(MPI_COMM_WORLD, &size);
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//
//     int *globaldata=NULL;
//     int localdata;
//
//     if (rank == 0) {
//         globaldata = malloc(size * sizeof(int) );
//         for (int i=0; i<size; i++)
//             globaldata[i] = 2*i+1;
//
//         printf("Processor %d has data: ", rank);
//         for (int i=0; i<size; i++)
//             printf("%d ", globaldata[i]);
//         printf("\n");
//     }
//
//     MPI_Scatter(globaldata, 1, MPI_INT, &localdata, 1, MPI_INT, 0, MPI_COMM_WORLD);
//
//     printf("Processor %d has data %d\n", rank, localdata);
//     localdata *= 2;
//     printf("Processor %d doubling the data, now has %d\n", rank, localdata);
//
//     MPI_Gather(&localdata, 1, MPI_INT, globaldata, 1, MPI_INT, 0, MPI_COMM_WORLD);
//
//     if (rank == 0) {
//         printf("Processor %d has data: ", rank);
//         for (int i=0; i<size; i++)
//             printf("%d ", globaldata[i]);
//         printf("\n");
//     }
//
//     if (rank == 0)
//         free(globaldata);
//
//     MPI_Finalize();
//     return 0;
// }
