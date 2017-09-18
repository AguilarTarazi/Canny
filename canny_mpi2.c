#include <omp.h>
#include <mpi.h>
#include <unistd.h>
/*
"Canny" edge detector code:
---------------------------

This text file contains the source code for a "Canny" edge detector. It
was written by Mike Heath (heath@csee.usf.edu) using some pieces of a
Canny edge detector originally written by someone at Michigan State
University.

There are three 'C' source code files in this text file. They are named
"canny_edge.c", "hysteresis.c" and "pgm_io.c". They were written and compiled
under SunOS 4.1.3. Since then they have also been compiled under Solaris.
To make an executable program: (1) Separate this file into three files with
the previously specified names, and then (2) compile the code using

gcc -o canny_edge canny_edge.c hysteresis.c pgm_io.c -lm
(Note: You can also use optimization such as -O3)

The resulting program, canny_edge, will process images in the PGM format.
Parameter selection is left up to the user. A broad range of parameters to
use as a starting point are: sigma 0.60-2.40, tlow 0.20-0.50 and,
thigh 0.60-0.90.

If you are using a Unix system, PGM file format conversion tools can be found
at ftp://wuarchive.wustl.edu/graphics/graphics/packages/pbmplus/.
Otherwise, it would be easy for anyone to rewrite the image I/O procedures
because they are listed in the separate file pgm_io.c.

If you want to check your compiled code, you can download grey-scale and edge
images from http://marathon.csee.usf.edu/edge/edge_detection.html. You can use
the parameters given in the edge filenames and check whether the edges that
are output from your program match the edge images posted at that address.

Mike Heath
(10/29/96)
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define VERBOSE 0
#define BOOSTBLURFACTOR 90.0

int read_pgm_image(char *infilename, unsigned char **image, int *rows, int *cols);
int write_pgm_image(char *outfilename, unsigned char *image, int rows, int cols, char *comment, int maxval);
void canny(unsigned char *image, int rows, int cols, float sigma, float tlow, float thigh, unsigned char **edge, char *fname);
void gaussian_smooth(unsigned char *image, int rows, int cols, float sigma,short int **smoothedim);
void make_gaussian_kernel(float sigma, float **kernel, int *windowsize);
void derrivative_x_y(short int *smoothedim, int rows, int cols,short int **delta_x, short int **delta_y);
void magnitude_x_y(short int *delta_x, short int *delta_y, int rows, int cols,short int **magnitude);
void apply_hysteresis(short int *mag, unsigned char *nms, int rows, int cols,float tlow, float thigh, unsigned char *edge);
void radian_direction(short int *delta_x, short int *delta_y, int rows,int cols, float **dir_radians, int xdirtag, int ydirtag);
double angle_radians(double x, double y);
void non_max_supp(short *mag, short *gradx, short *grady, int nrows, int ncols, unsigned char *result);

int rank = -1, numtasks;
int main(int argc, char *argv[])
{
    if(rank==0){
        printf("\n********************************************************\n");
        printf("********************* CANNY SERIAL *********************\n");
        printf("********************************************************\n");
    }
    double ini = MPI_Wtime();
    char *infilename = NULL; /* Name of the input image */
    char *dirfilename = NULL; /* Name of the output gradient direction image */
    char outfilename[128]; /* Name of the output "edge" image */
    char composedfname[128]; /* Name of the output "direction" image */
    unsigned char *image; /* The input image */
    unsigned char *edge; /* The output edge image */
    int rows, cols;      /* The dimensions of the image. */
    float sigma,         /* Standard deviation of the gaussian kernel. */
    tlow,    /* Fraction of the high threshold in hysteresis. */
    thigh;   /* High hysteresis threshold control. The actual
    threshold is the (100 * thigh) percentage point
    in the histogram of the magnitude of the
    gradient image that passes non-maximal
    suppression. */

    /****************************************************************************
    * Get the command line arguments.
    ****************************************************************************/
    if(argc < 5) {
        fprintf(stderr,"\n<USAGE> %s image sigma tlow thigh [writedirim]\n",argv[0]);
        fprintf(stderr,"\n      image:      An image to process. Must be in ");
        fprintf(stderr,"PGM format.\n");
        fprintf(stderr,"      sigma:      Standard deviation of the gaussian");
        fprintf(stderr," blur kernel.\n");
        fprintf(stderr,"      tlow:       Fraction (0.0-1.0) of the high ");
        fprintf(stderr,"edge strength threshold.\n");
        fprintf(stderr,"      thigh:      Fraction (0.0-1.0) of the distribution");
        fprintf(stderr," of non-zero edge\n                  strengths for ");
        fprintf(stderr,"hysteresis. The fraction is used to compute\n");
        fprintf(stderr,"                  the high edge strength threshold.\n");
        fprintf(stderr,"      writedirim: Optional argument to output ");
        fprintf(stderr,"a floating point");
        fprintf(stderr," direction image.\n\n");
        exit(1);
    }

    // CONFIGURACIONES MPI
    int dest, source, rc, count, tag=1;
    char inmsg, outmsg='x';
    MPI_Status Stat;

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Asigna el numero de hilo
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);


    infilename = argv[1];
    sigma = atof(argv[2]);
    tlow = atof(argv[3]);
    thigh = atof(argv[4]);

    if(rank == 0){
        if(argc == 6) dirfilename = infilename;
        else dirfilename = NULL;

        /****************************************************************************
        * Read in the image. This read function allocates memory for the image.
        ****************************************************************************/
        if(VERBOSE) printf("Reading the image %s.\n", infilename);
        if(read_pgm_image(infilename, &image, &rows, &cols) == 0) {
            fprintf(stderr, "Error reading the input image, %s.\n", infilename);
            exit(1);
        }

        /****************************************************************************
        * Perform the edge detection. All of the work takes place here.
        ****************************************************************************/
        if(VERBOSE) printf("Starting Canny edge detection.\n");
        if(dirfilename != NULL) {
            sprintf(composedfname, "%s_s_%3.2f_l_%3.2f_h_%3.2f.fim", infilename,
            sigma, tlow, thigh);
            dirfilename = composedfname;
        }
    }

    // Envío tamaño de la imagen a demás procesos
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    canny(image, rows, cols, sigma, tlow, thigh, &edge, dirfilename);

    if(rank == 0){
        /****************************************************************************
        * Write out the edge image to a file.
        ****************************************************************************/
        sprintf(outfilename, "imagen_mpi.pgm");
        // sprintf(outfilename, "%s_s_%3.2f_l_%3.2f_h_%3.2f.pgm", infilename, sigma, tlow, thigh);
        if(VERBOSE) printf("Writing the edge iname in the file %s.\n", outfilename);
        if(write_pgm_image(outfilename, edge, rows, cols, "", 255) == 0) {
            fprintf(stderr, "Error writing the edge image, %s.\n", outfilename);
            exit(1);
        }
        free(image);
    }
    MPI_Finalize();
    double fin = MPI_Wtime();
    if(rank==0){
        printf("\nTiempo total del programa: %.5g segundos\n\n", fin-ini);
    }
    return 0;
}

/*******************************************************************************
* PROCEDURE: canny
* PURPOSE: To perform canny edge detection.
* NAME: Mike Heath
* DATE: 2/15/96
*******************************************************************************/
void canny(unsigned char *image, int rows, int cols, float sigma,
    float tlow, float thigh, unsigned char **edge, char *fname)
    {
        FILE *fpdir=NULL;     /* File to write the gradient image to.     */
        unsigned char *nms;   /* Points that are local maximal magnitude. */
        short int *smoothedim, /* The image after gaussian smoothing.      */
        *delta_x,   /* The first devivative image, x-direction. */
        *delta_y,   /* The first derivative image, y-direction. */
        *magnitude; /* The magnitude of the gadient image.      */
        int r, c, pos;
        float *dir_radians=NULL; /* Gradient direction image.                */

        /****************************************************************************
        * Perform gaussian smoothing on the image using the input standard
        * deviation.
        ****************************************************************************/
        if(VERBOSE) printf("Smoothing the image using a gaussian kernel.\n");
        gaussian_smooth(image, rows, cols, sigma, &smoothedim);

        //   printf("SALIMOS DEL GAUSSIAN -> %d\n",rank );
        //   MPI_Barrier(MPI_COMM_WORLD);
        //   printf("BARRIER -> %d\n",rank );
        /****************************************************************************
        * Compute the first derivative in the x and y directions.
        ****************************************************************************/
        //   MPI_Bcast(&smoothedim, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if(VERBOSE) printf("Computing the X and Y first derivatives.\n");
        //   printf("ARRANCO DERRIVATIVE -> %d\n",rank );
        derrivative_x_y(smoothedim, rows, cols, &delta_x, &delta_y);
        if(rank == 0){

            /****************************************************************************
            * This option to write out the direction of the edge gradient was added
            * to make the information available for computing an edge quality figure
            * of merit.
            ****************************************************************************/
            if(fname != NULL) {
                /*************************************************************************
                * Compute the direction up the gradient, in radians that are
                * specified counteclockwise from the positive x-axis.
                *************************************************************************/
                radian_direction(delta_x, delta_y, rows, cols, &dir_radians, -1, -1);

                /*************************************************************************
                * Write the gradient direction image out to a file.
                *************************************************************************/
                if((fpdir = fopen(fname, "wb")) == NULL) {
                    fprintf(stderr, "Error opening the file %s for writing.\n", fname);
                    exit(1);
                }
                fwrite(dir_radians, sizeof(float), rows*cols, fpdir);
                fclose(fpdir);
                free(dir_radians);
            }

            /****************************************************************************
            * Compute the magnitude of the gradient.
            ****************************************************************************/
            if(VERBOSE) printf("Computing the magnitude of the gradient.\n");
            magnitude_x_y(delta_x, delta_y, rows, cols, &magnitude);

            /****************************************************************************
            * Perform non-maximal suppression.
            ****************************************************************************/
            if(VERBOSE) printf("Doing the non-maximal suppression.\n");
            if((nms = (unsigned char *) calloc(rows*cols,sizeof(unsigned char)))==NULL) {
                fprintf(stderr, "Error allocating the nms image.\n");
                exit(1);
            }

            non_max_supp(magnitude, delta_x, delta_y, rows, cols, nms);

            /****************************************************************************
            * Use hysteresis to mark the edge pixels.
            ****************************************************************************/
            if(VERBOSE) printf("Doing hysteresis thresholding.\n");
            if((*edge=(unsigned char *)calloc(rows*cols,sizeof(unsigned char))) ==NULL) {
                fprintf(stderr, "Error allocating the edge image.\n");
                exit(1);
            }

            apply_hysteresis(magnitude, nms, rows, cols, tlow, thigh, *edge);

            /****************************************************************************
            * Free all of the memory that we allocated except for the edge image that
            * is still being used to store out result.
            ****************************************************************************/
            //   free(smoothedim);
            // free(delta_x);
            // free(delta_y);
            free(magnitude);
            free(nms);
        }
    }

    /*******************************************************************************
    * Procedure: radian_direction
    * Purpose: To compute a direction of the gradient image from component dx and
    * dy images. Because not all derriviatives are computed in the same way, this
    * code allows for dx or dy to have been calculated in different ways.
    *
    * FOR X:  xdirtag = -1  for  [-1 0  1]
    *         xdirtag =  1  for  [ 1 0 -1]
    *
    * FOR Y:  ydirtag = -1  for  [-1 0  1]'
    *         ydirtag =  1  for  [ 1 0 -1]'
    *
    * The resulting angle is in radians measured counterclockwise from the
    * xdirection. The angle points "up the gradient".
    *******************************************************************************/
void radian_direction(short int *delta_x, short int *delta_y, int rows,int cols, float **dir_radians, int xdirtag, int ydirtag)
{
    int r, c, pos;
    float *dirim=NULL;
    double dx, dy;

    /****************************************************************************
    * Allocate an image to store the direction of the gradient.
    ****************************************************************************/
    if((dirim = (float *) calloc(rows*cols, sizeof(float))) == NULL) {
        fprintf(stderr, "Error allocating the gradient direction image.\n");
        exit(1);
    }
    *dir_radians = dirim;

    for(r=0,pos=0; r<rows; r++) {
        for(c=0; c<cols; c++,pos++) {
            dx = (double)delta_x[pos];
            dy = (double)delta_y[pos];

            if(xdirtag == 1) dx = -dx;
            if(ydirtag == -1) dy = -dy;

            dirim[pos] = (float)angle_radians(dx, dy);
        }
    }
}

/*******************************************************************************
* FUNCTION: angle_radians
* PURPOSE: This procedure computes the angle of a vector with components x and
* y. It returns this angle in radians with the answer being in the range
* 0 <= angle <2*PI.
*******************************************************************************/
double angle_radians(double x, double y)
{
    double xu, yu, ang;

    xu = fabs(x);
    yu = fabs(y);

    if((xu == 0) && (yu == 0)) return(0);

    ang = atan(yu/xu);

    if(x >= 0) {
        if(y >= 0) return(ang);
        else return(2*M_PI - ang);
    }
    else{
        if(y >= 0) return(M_PI - ang);
        else return(M_PI + ang);
    }
}

/*******************************************************************************
* PROCEDURE: magnitude_x_y
* PURPOSE: Compute the magnitude of the gradient. This is the square root of
* the sum of the squared derivative values.
* NAME: Mike Heath
* DATE: 2/15/96
*******************************************************************************/
void magnitude_x_y(short int *delta_x, short int *delta_y, int rows, int cols,short int **magnitude)
{
    int r, c, pos, sq1, sq2;

    /****************************************************************************
    * Allocate an image to store the magnitude of the gradient.
    ****************************************************************************/
    if((*magnitude = (short *) calloc(rows*cols, sizeof(short))) == NULL) {
        fprintf(stderr, "Error allocating the magnitude image.\n");
        exit(1);
    }

    for(r=0,pos=0; r<rows; r++) {
        for(c=0; c<cols; c++,pos++) {
            sq1 = (int)delta_x[pos] * (int)delta_x[pos];
            sq2 = (int)delta_y[pos] * (int)delta_y[pos];
            (*magnitude)[pos] = (short)(0.5 + sqrt((float)sq1 + (float)sq2));
            // if((*magnitude)[pos] >= 32768 || (*magnitude)[pos] < 0)
            //       printf("MAGNITUDE %d ----------------------------------------------------------\n", (*magnitude)[pos]);
        }
    }

}

/*******************************************************************************
* PROCEDURE: derrivative_x_y
* PURPOSE: Compute the first derivative of the image in both the x any y
* directions. The differential filters that are used are:
*
*                                          -1
*         dx =  -1 0 +1     and       dy =  0
*                                          +1
*
* NAME: Mike Heath
* DATE: 2/15/96
*******************************************************************************/
void derrivative_x_y(short int *smoothedim, int rows, int cols,
short int **delta_x, short int **delta_y)
{
    // printf("\n");
    int r, c, pos;
    double inicio, fin, inicioX, finX, inicioY, finY;
    double ini_derivate,fin_derivate;
    short int *delta_x_temp, *delta_y_temp;
    short int *smoothedim_temp, *p_ini, *p_fin;
    MPI_Status estado;
    ini_derivate = MPI_Wtime();

    int elementos, x, cantidad;
    elementos = rows*cols;
    x = 0;
    if(rows % numtasks != 0) x = numtasks - (rows % numtasks);
    cantidad = (rows + x)*cols;
    // printf("ROWS: %d --- COLS: %d --- CANTIDAD: %d\n",rows,cols,cantidad );
    /****************************************************************************
    * Allocate temporary buffer images to store the derivatives.
    ****************************************************************************/
    if(((delta_x_temp) = (short *) calloc(cantidad/numtasks, sizeof(short))) == NULL){
        fprintf(stderr, "Error allocating the delta_x image.\n");
        exit(1);
    }
    if(((delta_y_temp) = (short *) calloc(cantidad/numtasks, sizeof(short))) == NULL){
        fprintf(stderr, "Error allocating the delta_x image.\n");
        exit(1);
    }
    if(((p_ini) = (short *) calloc((cantidad/numtasks)+(2*cols), sizeof(short))) == NULL){
        fprintf(stderr, "Error allocating the delta_x image.\n");
        exit(1);
    }
    smoothedim_temp = &p_ini[cols];
    p_fin = &smoothedim_temp[cantidad/numtasks];

    /****************************************************************************
    * Allocate images to store the derivatives.
    ****************************************************************************/
    if(((*delta_x) = (short *) calloc(rows*cols, sizeof(short))) == NULL) {
        fprintf(stderr, "Error allocating the delta_x image.\n");
        exit(1);
    }
    if(((*delta_y) = (short *) calloc(rows*cols, sizeof(short))) == NULL) {
        fprintf(stderr, "Error allocating the delta_x image.\n");
        exit(1);
    }

    /****************************************************************************
    * Compute the x-derivative. Adjust the derivative at the borders to avoid
    * losing pixels.
    ****************************************************************************/
    // int counts[numtasks], displs[numtasks];
    // for (int t = 0; t < numtasks; t++) {
    //     counts[t] = cantidad / numtasks + 2*cols;
    //     displs[t] = t * cantidad / numtasks;
    //     // Para todos los procesos distintos del primero, retrocedo una fila,
    //     // para que quede una fila adicional arriba y una abajo
    //     if(rank > 0) displs[t] -= cols;
    //     printf("displs[%d] = %d\n",t, displs[t] );
    // }
    // MPI_Scatterv(smoothedim,counts,displs,MPI_SHORT,smoothedim_temp,cantidad/numtasks+2*cols,MPI_SHORT,0,MPI_COMM_WORLD);
    // printf("RANK %d cruzo Scatterv\n",rank );
    // if(rank > 0) // && rank != numtasks-1)
    // nms_temp = nms_temp + cols;

    if(VERBOSE) printf("   Computing the X-direction derivative.\n");
    MPI_Scatter(smoothedim, cantidad/numtasks, MPI_SHORT, smoothedim_temp, cantidad/numtasks, MPI_SHORT, 0, MPI_COMM_WORLD);
    int i,j;
    // for (i = 0; i < cantidad/numtasks; i=i+cols) {
    //     j=0;
    //     delta_x_temp[i+j] = smoothedim_temp[i+j+1] - smoothedim_temp[i+j];
    //     for (j = 1; j < cols-1; j++) {
    //         (delta_x_temp)[i+j] = smoothedim_temp[i+j+1] - smoothedim_temp[i+j-1];
    //     }
    //     delta_x_temp[i+j] = smoothedim_temp[i+j] - smoothedim_temp[i+j-1];
    // }

    // MPI_Gatherv(delta_x_temp,cantidad/numtasks,MPI_SHORT,*delta_x,counts,displs,MPI_SHORT,0,MPI_COMM_WORLD);
    // MPI_Gather(delta_x_temp,cantidad/numtasks,MPI_SHORT,*delta_x,cantidad/numtasks,MPI_SHORT,0,MPI_COMM_WORLD);
    // MPI_Allgather(delta_x_temp,cantidad/numtasks,MPI_SHORT,*delta_x,cantidad/numtasks,MPI_SHORT,MPI_COMM_WORLD);
    // if(rank==1) for(int i=0;i<10;i++) printf("%d ",smoothedim_temp[i] ); printf("\n");

    /****************************************************************************
    * Compute the y-derivative. Adjust the derivative at the borders to avoid
    * losing pixels.
    ****************************************************************************/
    // Se realizan los envios de las filas correspondientes para luego calcular la derivada de cada elemento
    // MPI_Barrier(MPI_COMM_WORLD);
    if(numtasks>1){
        if(rank==0){
            //Envia ultima fila al siguiente.//Recibe primera fila del siguiente.
            // printf("A. Soy %d\n",rank );
            MPI_Send(smoothedim_temp+((cantidad/numtasks)-cols),cols, MPI_SHORT, rank+1, 0, MPI_COMM_WORLD);
            MPI_Recv(p_fin, cols, MPI_SHORT,rank+1, 0, MPI_COMM_WORLD, &estado);
            // for(int i=0;i<1;i++){
            //     printf("[%d] send SMOO a [%d]\n",rank,rank+1 );
            //     for(int j=0;j<10;j++){
            //         printf("[%d]%d ",rank,smoothedim_temp[(cantidad/numtasks)-cols+j] );
            //     }
            //     printf("\n");
            //     printf("[%d] recv en P_FIN de [%d]\n",rank,rank+1 );
            //     for(int j=0;j<10;j++){
            //         printf("[%d]%d ",rank,p_fin[j] );
            //     }
            //     printf("\n");
            // }
        }
        else if(rank<numtasks-1){
            //Envia primera fila al anterior, y ultima fila al siguiente.
            //Recibe primera fila del siguiente, y ultima fila del anterior.
            // printf("B. Soy %d\n",rank );

            MPI_Recv(p_ini, cols, MPI_SHORT,rank-1, 0, MPI_COMM_WORLD, &estado);
            MPI_Send(smoothedim_temp,cols, MPI_SHORT, rank-1, 0, MPI_COMM_WORLD);
            MPI_Send(smoothedim_temp+((cantidad/numtasks)-cols),cols, MPI_SHORT, rank+1, 0, MPI_COMM_WORLD);
            MPI_Recv(p_fin, cols, MPI_SHORT,rank+1, 0, MPI_COMM_WORLD, &estado);
            // for(int i=0;i<1;i++){
            //     printf("[%d] recv en P_INI de [%d]\n",rank,rank-1 );
            //     for(int j=0;j<10;j++){
            //         printf("[%d]%d ",rank,p_ini[j] );
            //     }
            //     printf("\n");
            //     printf("[%d] send SMOO a [%d]\n",rank,rank-1 );
            //     for(int j=0;j<10;j++){
            //         printf("[%d]%d ",rank,smoothedim_temp[j] );
            //     }
            //     printf("\n");
            //     printf("[%d] send SMOO a [%d]\n",rank,rank+1 );
            //     for(int j=0;j<10;j++){
            //         printf("[%d]%d ",rank,smoothedim_temp[(cantidad/numtasks)-cols+j] );
            //     }
            //     printf("\n");
            //     printf("[%d] redv en P_FIN de [%d]\n",rank,rank+1 );
            //     for(int j=0;j<10;j++){
            //         printf("[%d]%d ",rank,p_fin[j] );
            //     }
            // }
        }
        else{
            //Envia primera fila al anterior.//Recibe ultima fila del anterior.
            // printf("C. Soy %d\n",rank );
            MPI_Recv(p_ini, cols, MPI_SHORT,rank-1, 0, MPI_COMM_WORLD, &estado);
            MPI_Send(smoothedim_temp,cols, MPI_SHORT, rank-1, 0, MPI_COMM_WORLD);
            // for(int i=0;i<1;i++){
            //     printf("[%d] recv en P_INI de [%d]\n",rank,rank-1 );
            //     for(int j=0;j<10;j++){
            //         printf("[%d]%d ",rank,p_ini[j] );
            //     }
            //     printf("\n");
            //     printf("[%d] send SMOO a [%d]\n",rank,rank-1 );
            //     for(int j=0;j<10;j++){
            //         printf("[%d]%d ",rank,smoothedim_temp[j] );
            //     }
            //     printf("\n");
            // }
        }

        // Se calcula la derivada de cada elemento
        if(rank==0){
            printf("D. Soy %d\n",rank );
            for(r=0;r<cols;r++)
                (delta_y_temp)[r] = smoothedim_temp[r+cols] - smoothedim_temp[r];
            for(r=cols;r<cantidad/numtasks;r++)
                (delta_y_temp)[r] = smoothedim_temp[r+cols] - smoothedim_temp[r-cols];
        }
        else if(rank==numtasks-1){
            printf("E. Soy %d\n",rank );
            for(r=0;r<cantidad/numtasks-cols;r++)
                (delta_y_temp)[r] = smoothedim_temp[r+cols] - smoothedim_temp[r-cols];
            for(r=cantidad/numtasks-cols;r<cantidad/numtasks;r++)
                (delta_y_temp)[r] = smoothedim_temp[r] - smoothedim_temp[r-cols];
        }
        else{
            printf("F. Soy %d\n",rank );
            for(r=0;r<cantidad/numtasks;r++){
                (delta_y_temp)[r] = smoothedim_temp[r+cols] - smoothedim_temp[r-cols];
            }
        }
        // MPI_Allgather(delta_y_temp,cantidad/numtasks,MPI_SHORT,*delta_y,cantidad/numtasks,MPI_SHORT,MPI_COMM_WORLD);
        MPI_Gather(delta_y_temp,cantidad/numtasks,MPI_SHORT,*delta_y,cantidad/numtasks,MPI_SHORT,0,MPI_COMM_WORLD);
        // free(delta_x_temp);
        // free(delta_y_temp);
    }
    else{
        // Si es un solo proceso, realiza el calculo directamente.
        printf("madafaca... \n");
        for(r=0;r<cols;r++)
        (*delta_y)[r] = smoothedim[r+cols] - smoothedim[r];
        for(r=(rows*cols)-cols;r<rows*cols;r++)
        (*delta_y)[r] = smoothedim[r] - smoothedim[r-cols];
        for(r=cols;r<(rows-1)*cols;r++)
        (*delta_y)[r] = smoothedim[r+cols] - smoothedim[r-cols];
    }

    fin_derivate = MPI_Wtime();
    if(rank==0){
        printf("======================DERITATIVE_X_Y====================\n");
        printf("Tiempo total funcion: \t\t\t%.5f	segundos\n", fin_derivate - ini_derivate);
        printf("========================================================\n\n");
    }
}

/*******************************************************************************
* PROCEDURE: gaussian_smooth
* PURPOSE: Blur an image with a gaussian filter.
* NAME: Mike Heath
* DATE: 2/15/96
*******************************************************************************/
void gaussian_smooth(unsigned char *image, int rows, int cols, float sigma, short int **smoothedim)
{
int r, c, rr, cc, /* Counter variables. */
windowsize,  /* Dimension of the gaussian kernel. */
center;      /* Half of the windowsize. */
float *tempim, *tempim_aux,  /* Buffer for separable filter gaussian smoothing. */
*kernel,   /* A one dimensional gaussian kernel. */
dot,       /* Dot product summing variable. */
sum;       /* Sum of the kernel weights variable. */
unsigned char *recvbuf;
clock_t start, end;
double time_in_seconds;
int elementos, x, cantidad;

elementos = rows*cols;
x = 0;
if(rows % numtasks != 0)
x = numtasks - (rows % numtasks);
cantidad = (rows + x)*cols;
//   printf("RANK %d: ELEMENTOS_REALES %d AGREGAR-X=%d, SOBRAN=%d  ELEMENTOS_TOTALES=%d,   ELEMENTOSxPROCESADOR=%d\n",rank,elementos,x,(elementos%numtasks),cantidad,cantidad/numtasks);

/****************************************************************************
* Allocate a temporary buffer image.
****************************************************************************/
if((recvbuf = (unsigned char *) calloc(cantidad/numtasks, sizeof(float))) == NULL) {
    printf("Error al inicializar recvbuf!!! en rank %d\n", rank );
    exit(1);
}

/****************************************************************************
* Allocate a temporary buffer image and the smoothed image.
****************************************************************************/
if((tempim_aux = (float *) calloc(cantidad/numtasks, sizeof(float))) == NULL) {
    fprintf(stderr, "Error allocating the buffer image.\n");
    exit(1);
}//else{
    //     printf("tempim_aux OK. %d lugares reservados.\n",(rows*cols)/numtasks);
    //   }

    /****************************************************************************
    * Create a 1-dimensional gaussian smoothing kernel.
    ****************************************************************************/
    if(VERBOSE) printf("   Computing the gaussian smoothing kernel.\n");
    make_gaussian_kernel(sigma, &kernel, &windowsize);
    // printf("RANK %d KERNEL %f SIGMA %f WINDOWSSIZE %d\n", rank, kernel[0], sigma, windowsize);
    center = windowsize / 2;

    if(rank == 0){
        /****************************************************************************
        * Allocate a temporary buffer image and the smoothed image.
        ****************************************************************************/
        if((tempim = (float *) calloc(cantidad, sizeof(float))) == NULL) {
            // if((tempim = (float *) calloc(rows*cols, sizeof(float))) == NULL) {
            fprintf(stderr, "Error allocating the buffer image.\n");
            exit(1);
        }
        if(((*smoothedim) = (short int *) calloc(cantidad, sizeof(short int))) == NULL) {
            // if(((*smoothedim) = (short int *) calloc(rows*cols, sizeof(short int))) == NULL) {
            fprintf(stderr, "Error allocating the smoothed image.\n");
            exit(1);
        }

        start = clock();

    }

    MPI_Scatter(image,cantidad/numtasks,MPI_UNSIGNED_CHAR,recvbuf,cantidad/numtasks,MPI_UNSIGNED_CHAR,0,MPI_COMM_WORLD);
    /****************************************************************************
    * Blur in the x - direction.
    ****************************************************************************/
    if(VERBOSE) printf("   Bluring the image in the X-direction.\n");
    // for(r=0; r<rows; r++) {
    //         for(c=0; c<cols; c++) {
    //                 dot = 0.0;
    //                 sum = 0.0;
    //                 for(cc=(-center); cc<=center; cc++) {
    //                         if( ((c+cc) >= 0) && ((c+cc) < cols) ) {
    //                                 // dot += (float)recvbuf[r*cols+(c+cc)] * kernel[center+cc];
    //                                 dot += (float)image[r*cols+(c+cc)] * kernel[center+cc];
    //                                 sum += kernel[center+cc];
    //                         }
    //                 }
    //                 // tempim_aux[r*cols+c] = dot/sum;
    //                 tempim[r*cols+c] = dot/sum;
    //         }
    // }

    // c=0; r=0;
    // for(int pos=0; pos<(rows*cols); pos++) {
    //       dot = 0.0;
    //       sum = 0.0;
    //       for(cc=(-center); cc<=center; cc++) {
    //               if( ((c+cc) >= 0) && ((c+cc) < cols) ) {
    //                       // dot += (float)recvbuf[r*cols+(c+cc)] * kernel[center+cc];
    //                       dot += (float)image[r*cols+(c+cc)] * kernel[center+cc];
    //                       sum += kernel[center+cc];
    //               }
    //       }
    //       // tempim_aux[r*cols+c] = dot/sum;
    //       tempim[r*cols+c] = dot/sum;
    //       c++;
    //       if(c == cols){
    //         r++;
    //         c = 0;
    //       }
    // }
    int cuenta = 0;

    // printf("(gaussian_smooth) rank %d\n",rank);
    c=0; r=0;
    // for(int pos=0; pos<(rows*cols)/numtasks; pos++) {
    for(int pos = 0; pos < cantidad / numtasks; pos++) {
        dot = 0.0;
        sum = 0.0;
        for(cc = (-center); cc <= center; cc++) {
            if( ((c+cc) >= 0) && ((c+cc) < cols) ) {
                dot += (float)recvbuf[r*cols+(c+cc)] * kernel[center+cc];
                sum += kernel[center+cc];
            }
        }
        tempim_aux[r*cols+c] = dot/sum;

        c++;
        if(c == cols){
            r++;
            c = 0;
        }
    }

    MPI_Gather (tempim_aux,cantidad/numtasks,MPI_FLOAT,tempim,cantidad/numtasks,MPI_FLOAT,0,MPI_COMM_WORLD);

    if(rank == 0){
        cuenta = 0;
        for (int p = 0; p < cantidad; p++) {
            // tempim[p] = tempim_aux[p];
            if(tempim[p] == 0)
            cuenta++;
            // else
            //       printf("-!- %lf    ",tempim[p]);
        }
        // printf("CUENTA! tempim: %d\n",cuenta);
        /****************************************************************************
        * Blur in the y - direction.
        ****************************************************************************/
        if(VERBOSE) printf("   Bluring the image in the Y-direction.\n");
        for(c=0; c<cols; c++) {
            for(r=0; r<rows; r++) {
                sum = 0.0;
                dot = 0.0;
                for(rr=(-center); rr<=center; rr++) {
                    if(((r+rr) >= 0) && ((r+rr) < rows)) {
                        dot += tempim[(r+rr)*cols+c] * kernel[center+rr];
                        // dot += tempim_aux[(r+rr)*cols+c] * kernel[center+rr];
                        sum += kernel[center+rr];
                    }
                }
                (*smoothedim)[r*cols+c] = (short int)(dot*BOOSTBLURFACTOR/sum + 0.5);
            }
        }
        end = clock();

        free(tempim);
        free(tempim_aux);
        free(kernel);

        time_in_seconds = (double)(end-start) / (double)CLOCKS_PER_SEC;
        printf("======================GAUSSIAN_SMOOTH===================\n");
        printf("Tiempo total funcion: \t\t\t%.5f	segundos\n", time_in_seconds);
        printf("========================================================\n\n");
    }
}

/*******************************************************************************
* PROCEDURE: make_gaussian_kernel
* PURPOSE: Create a one dimensional gaussian kernel.
* NAME: Mike Heath
* DATE: 2/15/96
*******************************************************************************/
void make_gaussian_kernel(float sigma, float **kernel, int *windowsize)
{
int i, center;
float x, fx, sum=0.0;

*windowsize = 1 + 2 * ceil(2.5 * sigma);
center = (*windowsize) / 2;

if(VERBOSE) printf("      The kernel has %d elements.\n", *windowsize);
if((*kernel = (float *) calloc((*windowsize), sizeof(float))) == NULL) {
    fprintf(stderr, "Error callocing the gaussian kernel array.\n");
    exit(1);
}

for(i=0; i<(*windowsize); i++) {
    x = (float)(i - center);
    fx = pow(2.71828, -0.5*x*x/(sigma*sigma)) / (sigma * sqrt(6.2831853));
    (*kernel)[i] = fx;
    sum += fx;
}

for(i=0; i<(*windowsize); i++) (*kernel)[i] /= sum;

if(VERBOSE) {
    printf("The filter coefficients are:\n");
    for(i=0; i<(*windowsize); i++)
    printf("kernel[%d] = %f\n", i, (*kernel)[i]);
}
}
//<------------------------- end canny_edge.c ------------------------->

//<------------------------- begin hysteresis.c ------------------------->
/*******************************************************************************
* FILE: hysteresis.c
* This code was re-written by Mike Heath from original code obtained indirectly
* from Michigan State University. heath@csee.usf.edu (Re-written in 1996).
*******************************************************************************/

#include <stdio.h>
#include <stdlib.h>

#define VERBOSE 0

#define NOEDGE 255
#define POSSIBLE_EDGE 128
#define EDGE 0

/*******************************************************************************
* PROCEDURE: follow_edges
* PURPOSE: This procedure edges is a recursive routine that traces edgs along
* all paths whose magnitude values remain above some specifyable lower
* threshhold.
* NAME: Mike Heath
* DATE: 2/15/96
*******************************************************************************/
void follow_edges(unsigned char *edgemapptr, short *edgemagptr, short lowval,int cols)
{
    short *tempmagptr;
    unsigned char *tempmapptr;
    int i;
    float thethresh;
    int x[8] = {1,1,0,-1,-1,-1,0,1},
    y[8] = {0,1,1,1,0,-1,-1,-1};

    for(i=0; i<8; i++) {
        tempmapptr = edgemapptr - y[i]*cols + x[i];
        tempmagptr = edgemagptr - y[i]*cols + x[i];

        if((*tempmapptr == POSSIBLE_EDGE) && (*tempmagptr > lowval)) {
            *tempmapptr = (unsigned char) EDGE;
            follow_edges(tempmapptr,tempmagptr, lowval, cols);
        }
    }
}

/*******************************************************************************
* PROCEDURE: apply_hysteresis
* PURPOSE: This routine finds edges that are above some high threshhold or
* are connected to a high pixel by a path of pixels greater than a low
* threshold.
* NAME: Mike Heath
* DATE: 2/15/96
*******************************************************************************/
void apply_hysteresis(short int *mag, unsigned char *nms, int rows, int cols,
float tlow, float thigh, unsigned char *edge)
{
    int r, c, pos, numedges, lowcount, highcount, lowthreshold, highthreshold,
    i, hist[32768], rr, cc;
    short int maximum_mag, sumpix;

    /****************************************************************************
    * Initialize the edge map to possible edges everywhere the non-maximal
    * suppression suggested there could be an edge except for the border. At
    * the border we say there can not be an edge because it makes the
    * follow_edges algorithm more efficient to not worry about tracking an
    * edge off the side of the image.
    ****************************************************************************/
    double inicio, fin, inicio1, fin1, inicio4, fin4, inicio5, fin5, inicio6, fin6, inicio7, fin7, inicio8, fin8;
    inicio = MPI_Wtime();

    int cant_threads = 4;

    inicio1 = MPI_Wtime();
    for(r=0,pos=0; r<rows; r++) {
        for(c=0; c<cols; c++,pos++) {
            if(nms[pos] == POSSIBLE_EDGE) edge[pos] = POSSIBLE_EDGE;
            else edge[pos] = NOEDGE;
        }
    }
    fin1 = MPI_Wtime();
    // printf("(apply_hysteresis) for1: %f segundos\n", (fin1-inicio1));

    for(r=0,pos=0; r<rows; r++,pos+=cols) {
        edge[pos] = NOEDGE;
        edge[pos+cols-1] = NOEDGE;
    }
    pos = (rows-1) * cols;
    for(c=0; c<cols; c++,pos++) {
        edge[c] = NOEDGE;
        edge[pos] = NOEDGE;
    }

    /****************************************************************************
    * Compute the histogram of the magnitude image. Then use the histogram to
    * compute hysteresis thresholds.
    ****************************************************************************/
    inicio4 = MPI_Wtime();
    for(r=0; r<32768; r++) hist[r] = 0;
    fin4 = MPI_Wtime();
    // printf("(apply_hysteresis) for4: %f segundos\n", (fin4-inicio4));

    inicio5 = MPI_Wtime();
    for(r=0,pos=0; r<rows; r++) {
        for(c=0; c<cols; c++,pos++) {
            if(edge[pos] == POSSIBLE_EDGE) hist[mag[pos]]++;
        }
    }
    fin5 = MPI_Wtime();
    // printf("(apply_hysteresis) for5: %f segundos\n", (fin5-inicio5));

    /****************************************************************************
    * Compute the number of pixels that passed the nonmaximal suppression.
    ****************************************************************************/
    inicio6 = MPI_Wtime();
    for(r=1,numedges=0; r<32768; r++) {
        if(hist[r] != 0) maximum_mag = r;
        numedges += hist[r];
    }
    fin6 = MPI_Wtime();
    // printf("(apply_hysteresis) for6: %f segundos\n", (fin6-inicio6));


    highcount = (int)(numedges * thigh + 0.5);

    /****************************************************************************
    * Compute the high threshold value as the (100 * thigh) percentage point
    * in the magnitude of the gradient histogram of all the pixels that passes
    * non-maximal suppression. Then calculate the low threshold as a fraction
    * of the computed high threshold value. John Canny said in his paper
    * "A Computational Approach to Edge Detection" that "The ratio of the
    * high to low threshold in the implementation is in the range two or three
    * to one." That means that in terms of this implementation, we should
    * choose tlow ~= 0.5 or 0.33333.
    ****************************************************************************/
    r = 1;
    numedges = hist[1];
    while((r<(maximum_mag-1)) && (numedges < highcount)) {
        r++;
        numedges += hist[r];
    }
    highthreshold = r;
    lowthreshold = (int)(highthreshold * tlow + 0.5);

    if(VERBOSE) {
        printf("The input low and high fractions of %f and %f computed to\n",
        tlow, thigh);
        printf("magnitude of the gradient threshold values of: %d %d\n",
        lowthreshold, highthreshold);
    }

    /****************************************************************************
    * This loop looks for pixels above the highthreshold to locate edges and
    * then calls follow_edges to continue the edge.
    ****************************************************************************/
    inicio7 = MPI_Wtime();
    for(r=0,pos=0; r<rows; r++) {
        for(c=0; c<cols; c++,pos++) {
            if((edge[pos] == POSSIBLE_EDGE) && (mag[pos] >= highthreshold)) {
                edge[pos] = EDGE;
                follow_edges((edge+pos), (mag+pos), lowthreshold, cols);
            }
        }
    }
    fin7 = MPI_Wtime();
    // printf("(apply_hysteresis) for7: %f segundos\n", (fin7-inicio7));

    /****************************************************************************
    * Set all the remaining possible edges to non-edges.
    ****************************************************************************/
    inicio8 = MPI_Wtime();
    // printf("despues inicio8\n");
    for(r=0,pos=0; r<rows; r++) {
        for(c=0; c<cols; c++,pos++) {
            if(edge[pos] != EDGE)
            edge[pos] = NOEDGE;
        }
    }
    // printf("antes fin8\n");
    fin8 = MPI_Wtime();
    // printf("(apply_hysteresis) for8: %f segundos\n", (fin8-inicio8));

    fin = MPI_Wtime();
    printf("=====================APPLY_HYSTERESIS===================\n");
    printf("Tiempo total funcion: \t\t\t%.5f	segundos\n", fin - inicio);
    printf("========================================================\n\n");
}

/*******************************************************************************
* PROCEDURE: non_max_supp
* PURPOSE: This routine applies non-maximal suppression to the magnitude of
* the gradient image.
* NAME: Mike Heath
* DATE: 2/15/96
*******************************************************************************/
void non_max_supp(short *mag, short *gradx, short *grady, int nrows, int ncols,unsigned char *result)
{
    int rowcount, colcount,count;
    short *magrowptr,*magptr;
    short *gxrowptr,*gxptr;
    short *gyrowptr,*gyptr,z1,z2;
    short m00,gx,gy;
    float mag1,mag2,xperp,yperp;
    unsigned char *resultrowptr, *resultptr;



    /****************************************************************************
    * Zero the edges of the result image.
    ****************************************************************************/
    for(count=0,resultrowptr=result,resultptr=result+ncols*(nrows-1);
    count<ncols; resultptr++,resultrowptr++,count++) {
        *resultrowptr = *resultptr = (unsigned char) 0;
    }

    for(count=0,resultptr=result,resultrowptr=result+ncols-1;
        count<nrows; count++,resultptr+=ncols,resultrowptr+=ncols) {
            *resultptr = *resultrowptr = (unsigned char) 0;
    }

    clock_t start, end;
    double time_in_seconds;
    start = clock();
    /****************************************************************************
    * Suppress non-maximum points.
    ****************************************************************************/
    magrowptr=mag+ncols+1,gxrowptr=gradx+ncols+1,gyrowptr=grady+ncols+1,resultrowptr=result+ncols+1;
    for(rowcount=1;rowcount<nrows-2;rowcount++) {
        for(colcount=1,magptr=magrowptr,gxptr=gxrowptr,gyptr=gyrowptr,
            resultptr=resultrowptr; colcount<ncols-2;
            colcount++,magptr++,gxptr++,gyptr++,resultptr++) {
            m00 = *magptr;
            if(m00 == 0) {
                *resultptr = (unsigned char) NOEDGE;
            }
            else{
                xperp = -(gx = *gxptr)/((float)m00);
                yperp = (gy = *gyptr)/((float)m00);
            }

            if(gx >= 0) {
                if(gy >= 0) {
                    if (gx >= gy)
                    {
                        /* 111 */
                        /* Left point */
                        z1 = *(magptr - 1);
                        z2 = *(magptr - ncols - 1);

                        mag1 = (m00 - z1)*xperp + (z2 - z1)*yperp;

                        /* Right point */
                        z1 = *(magptr + 1);
                        z2 = *(magptr + ncols + 1);

                        mag2 = (m00 - z1)*xperp + (z2 - z1)*yperp;
                    }
                    else
                    {
                        /* 110 */
                        /* Left point */
                        z1 = *(magptr - ncols);
                        z2 = *(magptr - ncols - 1);

                        mag1 = (z1 - z2)*xperp + (z1 - m00)*yperp;

                        /* Right point */
                        z1 = *(magptr + ncols);
                        z2 = *(magptr + ncols + 1);

                        mag2 = (z1 - z2)*xperp + (z1 - m00)*yperp;
                    }
                }
                else
                {
                    if (gx >= -gy)
                    {
                        /* 101 */
                        /* Left point */
                        z1 = *(magptr - 1);
                        z2 = *(magptr + ncols - 1);

                        mag1 = (m00 - z1)*xperp + (z1 - z2)*yperp;

                        /* Right point */
                        z1 = *(magptr + 1);
                        z2 = *(magptr - ncols + 1);

                        mag2 = (m00 - z1)*xperp + (z1 - z2)*yperp;
                    }
                    else
                    {
                        /* 100 */
                        /* Left point */
                        z1 = *(magptr + ncols);
                        z2 = *(magptr + ncols - 1);

                        mag1 = (z1 - z2)*xperp + (m00 - z1)*yperp;

                        /* Right point */
                        z1 = *(magptr - ncols);
                        z2 = *(magptr - ncols + 1);

                        mag2 = (z1 - z2)*xperp  + (m00 - z1)*yperp;
                    }
                }
            }
            else
            {
                if ((gy = *gyptr) >= 0)
                {
                    if (-gx >= gy)
                    {
                        /* 011 */
                        /* Left point */
                        z1 = *(magptr + 1);
                        z2 = *(magptr - ncols + 1);

                        mag1 = (z1 - m00)*xperp + (z2 - z1)*yperp;

                        /* Right point */
                        z1 = *(magptr - 1);
                        z2 = *(magptr + ncols - 1);

                        mag2 = (z1 - m00)*xperp + (z2 - z1)*yperp;
                    }
                    else
                    {
                        /* 010 */
                        /* Left point */
                        z1 = *(magptr - ncols);
                        z2 = *(magptr - ncols + 1);

                        mag1 = (z2 - z1)*xperp + (z1 - m00)*yperp;

                        /* Right point */
                        z1 = *(magptr + ncols);
                        z2 = *(magptr + ncols - 1);

                        mag2 = (z2 - z1)*xperp + (z1 - m00)*yperp;
                    }
                }
                else
                {
                    if (-gx > -gy)
                    {
                        /* 001 */
                        /* Left point */
                        z1 = *(magptr + 1);
                        z2 = *(magptr + ncols + 1);

                        mag1 = (z1 - m00)*xperp + (z1 - z2)*yperp;

                        /* Right point */
                        z1 = *(magptr - 1);
                        z2 = *(magptr - ncols - 1);

                        mag2 = (z1 - m00)*xperp + (z1 - z2)*yperp;
                    }
                    else
                    {
                        /* 000 */
                        /* Left point */
                        z1 = *(magptr + ncols);
                        z2 = *(magptr + ncols + 1);

                        mag1 = (z2 - z1)*xperp + (m00 - z1)*yperp;

                        /* Right point */
                        z1 = *(magptr - ncols);
                        z2 = *(magptr - ncols - 1);

                        mag2 = (z2 - z1)*xperp + (m00 - z1)*yperp;
                    }
                }
            }

            /* Now determine if the current point is a maximum point */

            if ((mag1 > 0.0) || (mag2 > 0.0))
            {
                *resultptr = (unsigned char) NOEDGE;
            }
            else
            {
                if (mag2 == 0.0)
                *resultptr = (unsigned char) NOEDGE;
                else
                *resultptr = (unsigned char) POSSIBLE_EDGE;
            }
        }
        magrowptr+=ncols,gyrowptr+=ncols,gxrowptr+=ncols,resultrowptr+=ncols;
    }
    end = clock();
    time_in_seconds = (double)(end-start) / (double)CLOCKS_PER_SEC;
    printf("=======================NON_MAX_SUPP====================\n");
    printf("Tiempo total funcion: \t\t\t%.5f	segundos\n", time_in_seconds);
    printf("========================================================\n\n");
    // printf("(non_max_supp) %.4g segundos\n", (time_in_seconds));
}
//<------------------------- end hysteresis.c ------------------------->

//<------------------------- begin pgm_io.c------------------------->
/*******************************************************************************
* FILE: pgm_io.c
* This code was written by Mike Heath. heath@csee.usf.edu (in 1995).
*******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/******************************************************************************
* Function: read_pgm_image
* Purpose: This function reads in an image in PGM format. The image can be
* read in from either a file or from standard input. The image is only read
* from standard input when infilename = NULL. Because the PGM format includes
* the number of columns and the number of rows in the image, these are read
* from the file. Memory to store the image is allocated in this function.
* All comments in the header are discarded in the process of reading the
* image. Upon failure, this function returns 0, upon sucess it returns 1.
******************************************************************************/
int read_pgm_image(char *infilename, unsigned char **image, int *rows,
    int *cols)
    {
        FILE *fp;
        char buf[71];

        /***************************************************************************
        * Open the input image file for reading if a filename was given. If no
        * filename was provided, set fp to read from standard input.
        ***************************************************************************/
        if(infilename == NULL) fp = stdin;
        else{
            if((fp = fopen(infilename, "r")) == NULL) {
                fprintf(stderr, "Error reading the file %s in read_pgm_image().\n",
                infilename);
                return(0);
            }
        }

        /***************************************************************************
        * Verify that the image is in PGM format, read in the number of columns
        * and rows in the image and scan past all of the header information.
        ***************************************************************************/
        fgets(buf, 70, fp);
        if(strncmp(buf,"P5",2) != 0) {
            fprintf(stderr, "The file %s is not in PGM format in ", infilename);
            fprintf(stderr, "read_pgm_image().\n");
            if(fp != stdin) fclose(fp);
            return(0);
        }
        do { fgets(buf, 70, fp); } while(buf[0] == '#'); /* skip all comment lines */
        sscanf(buf, "%d %d", cols, rows);
        do { fgets(buf, 70, fp); } while(buf[0] == '#'); /* skip all comment lines */

        /***************************************************************************
        * Allocate memory to store the image then read the image from the file.
        ***************************************************************************/
        if(((*image) = (unsigned char *) malloc((*rows)*(*cols))) == NULL) {
            fprintf(stderr, "Memory allocation failure in read_pgm_image().\n");
            if(fp != stdin) fclose(fp);
            return(0);
        }
        if((*rows) != fread((*image), (*cols), (*rows), fp)) {
            fprintf(stderr, "Error reading the image data in read_pgm_image().\n");
            if(fp != stdin) fclose(fp);
            free((*image));
            return(0);
        }

        if(fp != stdin) fclose(fp);
        return(1);
    }

    /******************************************************************************
    * Function: write_pgm_image
    * Purpose: This function writes an image in PGM format. The file is either
    * written to the file specified by outfilename or to standard output if
    * outfilename = NULL. A comment can be written to the header if coment != NULL.
    ******************************************************************************/
    int write_pgm_image(char *outfilename, unsigned char *image, int rows,
        int cols, char *comment, int maxval)
        {
            FILE *fp;

            /***************************************************************************
            * Open the output image file for writing if a filename was given. If no
            * filename was provided, set fp to write to standard output.
            ***************************************************************************/
            if(outfilename == NULL) fp = stdout;
            else{
                if((fp = fopen(outfilename, "w")) == NULL) {
                    fprintf(stderr, "Error writing the file %s in write_pgm_image().\n",
                    outfilename);
                    return(0);
                }
            }

            /***************************************************************************
            * Write the header information to the PGM file.
            ***************************************************************************/
            fprintf(fp, "P5\n%d %d\n", cols, rows);
            if(comment != NULL)
            if(strlen(comment) <= 70) fprintf(fp, "# %s\n", comment);
            fprintf(fp, "%d\n", maxval);

            /***************************************************************************
            * Write the image data to the file.
            ***************************************************************************/
            if(rows != fwrite(image, cols, rows, fp)) {
                fprintf(stderr, "Error writing the image data in write_pgm_image().\n");
                if(fp != stdout) fclose(fp);
                return(0);
            }

            if(fp != stdout) fclose(fp);
            return(1);
        }

        /******************************************************************************
        * Function: read_ppm_image
        * Purpose: This function reads in an image in PPM format. The image can be
        * read in from either a file or from standard input. The image is only read
        * from standard input when infilename = NULL. Because the PPM format includes
        * the number of columns and the number of rows in the image, these are read
        * from the file. Memory to store the image is allocated in this function.
        * All comments in the header are discarded in the process of reading the
        * image. Upon failure, this function returns 0, upon sucess it returns 1.
        ******************************************************************************/
        int read_ppm_image(char *infilename, unsigned char **image_red,
            unsigned char **image_grn, unsigned char **image_blu, int *rows,
            int *cols)
            {
                FILE *fp;
                char buf[71];
                int p, size;

                /***************************************************************************
                * Open the input image file for reading if a filename was given. If no
                * filename was provided, set fp to read from standard input.
                ***************************************************************************/
                if(infilename == NULL) fp = stdin;
                else{
                    if((fp = fopen(infilename, "r")) == NULL) {
                        fprintf(stderr, "Error reading the file %s in read_ppm_image().\n",
                        infilename);
                        return(0);
                    }
                }

                /***************************************************************************
                * Verify that the image is in PPM format, read in the number of columns
                * and rows in the image and scan past all of the header information.
                ***************************************************************************/
                fgets(buf, 70, fp);
                if(strncmp(buf,"P6",2) != 0) {
                    fprintf(stderr, "The file %s is not in PPM format in ", infilename);
                    fprintf(stderr, "read_ppm_image().\n");
                    if(fp != stdin) fclose(fp);
                    return(0);
                }
                do { fgets(buf, 70, fp); } while(buf[0] == '#'); /* skip all comment lines */
                sscanf(buf, "%d %d", cols, rows);
                do { fgets(buf, 70, fp); } while(buf[0] == '#'); /* skip all comment lines */

                /***************************************************************************
                * Allocate memory to store the image then read the image from the file.
                ***************************************************************************/
                if(((*image_red) = (unsigned char *) malloc((*rows)*(*cols))) == NULL) {
                    fprintf(stderr, "Memory allocation failure in read_ppm_image().\n");
                    if(fp != stdin) fclose(fp);
                    return(0);
                }
                if(((*image_grn) = (unsigned char *) malloc((*rows)*(*cols))) == NULL) {
                    fprintf(stderr, "Memory allocation failure in read_ppm_image().\n");
                    if(fp != stdin) fclose(fp);
                    return(0);
                }
                if(((*image_blu) = (unsigned char *) malloc((*rows)*(*cols))) == NULL) {
                    fprintf(stderr, "Memory allocation failure in read_ppm_image().\n");
                    if(fp != stdin) fclose(fp);
                    return(0);
                }

                size = (*rows)*(*cols);
                for(p=0; p<size; p++) {
                    (*image_red)[p] = (unsigned char)fgetc(fp);
                    (*image_grn)[p] = (unsigned char)fgetc(fp);
                    (*image_blu)[p] = (unsigned char)fgetc(fp);
                }

                if(fp != stdin) fclose(fp);
                return(1);
            }

            /******************************************************************************
            * Function: write_ppm_image
            * Purpose: This function writes an image in PPM format. The file is either
            * written to the file specified by outfilename or to standard output if
            * outfilename = NULL. A comment can be written to the header if coment != NULL.
            ******************************************************************************/
            int write_ppm_image(char *outfilename, unsigned char *image_red,
                unsigned char *image_grn, unsigned char *image_blu, int rows,
                int cols, char *comment, int maxval)
                {
                    FILE *fp;
                    long size, p;

                    /***************************************************************************
                    * Open the output image file for writing if a filename was given. If no
                    * filename was provided, set fp to write to standard output.
                    ***************************************************************************/
                    if(outfilename == NULL) fp = stdout;
                    else{
                        if((fp = fopen(outfilename, "w")) == NULL) {
                            fprintf(stderr, "Error writing the file %s in write_pgm_image().\n",
                            outfilename);
                            return(0);
                        }
                    }

                    /***************************************************************************
                    * Write the header information to the PGM file.
                    ***************************************************************************/
                    fprintf(fp, "P6\n%d %d\n", cols, rows);
                    if(comment != NULL)
                    if(strlen(comment) <= 70) fprintf(fp, "# %s\n", comment);
                    fprintf(fp, "%d\n", maxval);

                    /***************************************************************************
                    * Write the image data to the file.
                    ***************************************************************************/
                    size = (long)rows * (long)cols;
                    for(p=0; p<size; p++) { /* Write the image in pixel interleaved format. */
                        fputc(image_red[p], fp);
                        fputc(image_grn[p], fp);
                        fputc(image_blu[p], fp);
                    }

                    if(fp != stdout) fclose(fp);
                    return(1);
                }
                //<------------------------- end pgm_io.c ------------------------->
