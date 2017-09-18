
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
void follow_edges(unsigned char *edgemapptr, short *edgemagptr, short lowval,
                  int cols)
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
        // #pragma omp parallel
        // {
        //         printf(">>> Hilos %d\n", omp_get_num_threads());
        // }

        double inicio, fin, inicio1, fin1, inicio4, fin4, inicio5, fin5, inicio6, fin6, inicio7, fin7, inicio8, fin8;
        inicio = omp_get_wtime();

        int cant_threads = 4;
        inicio1 = omp_get_wtime();
        #pragma omp parallel shared(pos,edge,nms) num_threads(cant_threads)
        {
                #pragma omp for
                for(pos=0; pos<(rows*cols); pos++) {
                        if(nms[pos] == POSSIBLE_EDGE)
                                edge[pos] = POSSIBLE_EDGE;
                        else
                                edge[pos] = NOEDGE;
                }
        }
        fin1 = omp_get_wtime();
        printf("(apply_hysteresis) for1: %f segundos\n", (fin1-inicio1));

        //dos
        for(r=0,pos=0; r<rows; r++,pos+=cols) {
                edge[pos] = NOEDGE;
                edge[pos+cols-1] = NOEDGE;
        }
        pos = (rows-1) * cols;

        //tres
        for(c=0; c<cols; c++,pos++) {
                edge[c] = NOEDGE;
                edge[pos] = NOEDGE;
        }

        /****************************************************************************
        * Compute the histogram of the magnitude image. Then use the histogram to
        * compute hysteresis thresholds.
        ****************************************************************************/
        inicio4 = omp_get_wtime();
        #pragma omp paralell shared(hist) num_threads(cant_threads)
        {
          #pragma omp for
                for(r=0; r<32768; r++)
                        hist[r] = 0;
        }
        fin4 = omp_get_wtime();
        printf("(apply_hysteresis) for4: %f segundos\n", (fin4-inicio4));

        inicio5 = omp_get_wtime();
        #pragma omp paralel shared(edge,hist,mag) num_threads(cant_threads)
        {
                printf("(apply_hysteresis) for5: thread %d \n", omp_get_thread_num());
                #pragma omp for
                for(pos=0; pos<(rows*cols); pos++) {
                        if(edge[pos] == POSSIBLE_EDGE)
                                hist[mag[pos]]++;
                }
        }
        fin5 = omp_get_wtime();
        printf("(apply_hysteresis) for5: %f segundos\n", (fin5-inicio5));

        /****************************************************************************
        * Compute the number of pixels that passed the nonmaximal suppression.
        ****************************************************************************/
        inicio6 = omp_get_wtime();
        numedges=0;
        #pragma omp paralell shared(hist,numedges) num_threads(cant_threads)
        {
          #pragma omp for
                for(r=1; r<32768; r++) {
                        if(hist[r] != 0)
                                maximum_mag = r;
                        numedges += hist[r];
                }
        }
        fin6 = omp_get_wtime();
        printf("(apply_hysteresis) for6: %f segundos\n", (fin6-inicio6));

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
        inicio7 = omp_get_wtime();
        #pragma omp paralell shared(edge,mag) private(highthreshold, lowthreshold) num_threads(cant_threads)
        {
          #pragma omp for
                for(pos=0; pos<(rows*cols); pos++) {
                        if((edge[pos] == POSSIBLE_EDGE) && (mag[pos] >= highthreshold)) {
                                edge[pos] = EDGE;
                                follow_edges((edge+pos), (mag+pos), lowthreshold, cols);
                        }
                }
        }
        fin7 = omp_get_wtime();
        printf("(apply_hysteresis) for7: %f segundos\n", (fin7-inicio7));

        /****************************************************************************
        * Set all the remaining possible edges to non-edges.
        ****************************************************************************/
        inicio8 = omp_get_wtime();
        #pragma omp paralell shared(edge) num_threads(cant_threads)
        {
          #pragma omp for
                for(pos=0; pos<(rows*cols); pos++) {
                        if(edge[pos] != EDGE)
                                edge[pos] = NOEDGE;
                }
        }
        fin8 = omp_get_wtime();
        printf("(apply_hysteresis) for8: %f segundos\n", (fin8-inicio8));

        fin = omp_get_wtime();
        printf("(apply_hysteresis) %f segundos\n", (fin-inicio));
}


/*******************************************************************************
* PROCEDURE: non_max_supp
* PURPOSE: This routine applies non-maximal suppression to the magnitude of
* the gradient image.
* NAME: Mike Heath
* DATE: 2/15/96
*******************************************************************************/
void non_max_suppMODIFICADO(short *mag, short *gradx, short *grady, int nrows, int ncols, unsigned char *result)
{
        int rowcount, colcount,count;
        short *magrowptr,*magptr;
        short *gxrowptr,*gxptr;
        short *gyrowptr,*gyptr,z1,z2;
        short m00,gx,gy;
        float mag1,mag2,xperp,yperp;
        unsigned char *resultrowptr, *resultptr;

        // COPIAS
        short *magptr_copia, *gxptr_copia, *gyptr_copia;
        unsigned char *resultptr_copia;


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
        // start = clock();
        /****************************************************************************
        * Suppress non-maximum points.
        ****************************************************************************/
        magrowptr=mag+ncols+1; gxrowptr=gradx+ncols+1;
        gyrowptr=grady+ncols+1; resultrowptr=result+ncols+1;
        int cant_threads = 4;
        double inicio, fin;
        inicio = omp_get_wtime();

        int ncols_copia = ncols;
        int se_ejecuto = 0;
        for(
                rowcount=1;
                // magrowptr=mag+ncols+1,gxrowptr=gradx+ncols+1,
                // gyrowptr=grady+ncols+1,resultrowptr=result+ncols+1;
                rowcount<nrows-2;
                rowcount++
                // magrowptr+=ncols,gyrowptr+=ncols,gxrowptr+=ncols,
                // resultrowptr+=ncols
                ) {
                // if(omp_get_thread_num() == 0)
                //         printf("Soy el hilo %d y mi colcount es %d (ncols-2=%d)\n",omp_get_thread_num(),colcount,ncols-2);
                magptr=magrowptr; gxptr=gxrowptr; gyptr=gyrowptr; resultptr=resultrowptr;
                // printf("ncols %d\n", ncols);
                #pragma omp parallel shared(magptr,gxptr,gyptr,resultptr,ncols_copia) private(colcount,m00,z1,z2,mag1,mag2,gx,gy,xperp,yperp,magptr_copia,gxptr_copia,gyptr_copia,resultptr_copia,ncols) num_threads(cant_threads)
                {
                        // printf("\n---\n");
                        #pragma omp for
                        for (colcount=1;
                             // magptr=magrowptr; gxptr=gxrowptr; gyptr=gyrowptr; resultptr=resultrowptr;
                             colcount<ncols_copia-2;
                             //  colcount<ncols_copia-2;
                             colcount++
                             ) // ,magptr++,gxptr++,gyptr++,resultptr++
                        {
                                // printf("-%d.%d- ",omp_get_thread_num(),ncols);
                                // printf("-5-");
                                se_ejecuto = 1;
                                // #pragma omp critical (copias)
                                // {
                                //         magptr_copia = magptr++;
                                //         gxptr_copia = gxptr++;
                                //         gyptr_copia = gyptr++;
                                //         resultptr_copia = resultptr++;
                                // }
                                #pragma omp critical (copias)
                                {
                                        magptr_copia = magptr;
                                        gxptr_copia = gxptr;
                                        gyptr_copia = gyptr;
                                        resultptr_copia = resultptr;
                                        magptr++; gxptr++; gyptr++; resultptr++;
                                }

                                m00 = *magptr_copia;
                                if(m00 == 0) {
                                        *resultptr_copia = (unsigned char) NOEDGE;
                                }
                                else{
                                        xperp = -(gx = *gxptr_copia)/((float)m00);
                                        yperp = (gy = *gyptr_copia)/((float)m00);
                                }

                                if(gx >= 0) {
                                        if(gy >= 0) {
                                                if (gx >= gy)
                                                {
                                                        /* 111 */
                                                        /* Left point */
                                                        z1 = *(magptr_copia - 1);
                                                        z2 = *(magptr_copia - ncols - 1);

                                                        mag1 = (m00 - z1)*xperp + (z2 - z1)*yperp;

                                                        /* Right point */
                                                        z1 = *(magptr_copia + 1);
                                                        z2 = *(magptr_copia + ncols + 1);

                                                        mag2 = (m00 - z1)*xperp + (z2 - z1)*yperp;
                                                }
                                                else
                                                {
                                                        /* 110 */
                                                        /* Left point */
                                                        z1 = *(magptr_copia - ncols);
                                                        z2 = *(magptr_copia - ncols - 1);

                                                        mag1 = (z1 - z2)*xperp + (z1 - m00)*yperp;

                                                        /* Right point */
                                                        z1 = *(magptr_copia + ncols);
                                                        z2 = *(magptr_copia + ncols + 1);

                                                        mag2 = (z1 - z2)*xperp + (z1 - m00)*yperp;
                                                }
                                        }
                                        else
                                        {
                                                if (gx >= -gy)
                                                {
                                                        /* 101 */
                                                        /* Left point */
                                                        z1 = *(magptr_copia - 1);
                                                        z2 = *(magptr_copia + ncols - 1);

                                                        mag1 = (m00 - z1)*xperp + (z1 - z2)*yperp;

                                                        /* Right point */
                                                        z1 = *(magptr_copia + 1);
                                                        z2 = *(magptr_copia - ncols + 1);

                                                        mag2 = (m00 - z1)*xperp + (z1 - z2)*yperp;
                                                }
                                                else
                                                {
                                                        /* 100 */
                                                        /* Left point */
                                                        z1 = *(magptr_copia + ncols);
                                                        z2 = *(magptr_copia + ncols - 1);

                                                        mag1 = (z1 - z2)*xperp + (m00 - z1)*yperp;

                                                        /* Right point */
                                                        z1 = *(magptr_copia - ncols);
                                                        z2 = *(magptr_copia - ncols + 1);

                                                        mag2 = (z1 - z2)*xperp  + (m00 - z1)*yperp;
                                                }
                                        }
                                }
                                else {
                                        if ((gy = *gyptr_copia) >= 0)
                                        {
                                                if (-gx >= gy)
                                                {
                                                        /* 011 */
                                                        /* Left point */
                                                        z1 = *(magptr_copia + 1);
                                                        z2 = *(magptr_copia - ncols + 1);

                                                        mag1 = (z1 - m00)*xperp + (z2 - z1)*yperp;

                                                        /* Right point */
                                                        z1 = *(magptr_copia - 1);
                                                        z2 = *(magptr_copia + ncols - 1);

                                                        mag2 = (z1 - m00)*xperp + (z2 - z1)*yperp;
                                                }
                                                else
                                                {
                                                        /* 010 */
                                                        /* Left point */
                                                        z1 = *(magptr_copia - ncols);
                                                        z2 = *(magptr_copia - ncols + 1);

                                                        mag1 = (z2 - z1)*xperp + (z1 - m00)*yperp;

                                                        /* Right point */
                                                        z1 = *(magptr_copia + ncols);
                                                        z2 = *(magptr_copia + ncols - 1);

                                                        mag2 = (z2 - z1)*xperp + (z1 - m00)*yperp;
                                                }
                                        }
                                        else
                                        {
                                                if (-gx > -gy)
                                                {
                                                        /* 001 */
                                                        /* Left point */
                                                        z1 = *(magptr_copia + 1);
                                                        z2 = *(magptr_copia + ncols + 1);

                                                        mag1 = (z1 - m00)*xperp + (z1 - z2)*yperp;

                                                        /* Right point */
                                                        z1 = *(magptr_copia - 1);
                                                        z2 = *(magptr_copia - ncols - 1);

                                                        mag2 = (z1 - m00)*xperp + (z1 - z2)*yperp;
                                                }
                                                else
                                                {
                                                        /* 000 */
                                                        /* Left point */
                                                        z1 = *(magptr_copia + ncols);
                                                        z2 = *(magptr_copia + ncols + 1);

                                                        mag1 = (z2 - z1)*xperp + (m00 - z1)*yperp;

                                                        /* Right point */
                                                        z1 = *(magptr_copia - ncols);
                                                        z2 = *(magptr_copia - ncols - 1);

                                                        mag2 = (z2 - z1)*xperp + (m00 - z1)*yperp;
                                                }
                                        }
                                }

                                /* Now determine if the current point is a maximum point */

                                if ((mag1 > 0.0) || (mag2 > 0.0))
                                {
                                        *resultptr_copia = (unsigned char) NOEDGE;
                                }
                                else
                                {
                                        if (mag2 == 0.0)
                                                *resultptr_copia = (unsigned char) NOEDGE;
                                        else
                                                *resultptr_copia = (unsigned char) POSSIBLE_EDGE;
                                }

                        } // FIN FOR INTERNO

                }     // FIN PRAGMA PARALLEL
                // #pragma omp barrier
                magrowptr+=ncols_copia; gyrowptr+=ncols_copia; gxrowptr+=ncols_copia; resultrowptr+=ncols_copia;
        }
        fin = omp_get_wtime();
        printf("\n(non_max_supp) TIEMPO PARALELO %f segundos\n", (fin-inicio));
        if(se_ejecuto)
                printf("Se ejecuto el for interno\n");
        else
                printf("No se ejecuto el for interno\n");
}
//<------------------------- end hysteresis.c ------------------------->


/*******************************************************************************
* PROCEDURE: non_max_supp
* PURPOSE: This routine applies non-maximal suppression to the magnitude of
* the gradient image.
* NAME: Mike Heath
* DATE: 2/15/96
*******************************************************************************/
void non_max_supp(short *mag, short *gradx, short *grady, int nrows, int ncols,
                  unsigned char *result)
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

        /****************************************************************************
        * Suppress non-maximum points.
        ****************************************************************************/
        for(rowcount=1,magrowptr=mag+ncols+1,gxrowptr=gradx+ncols+1,
            gyrowptr=grady+ncols+1,resultrowptr=result+ncols+1;
            rowcount<nrows-2;
            rowcount++,magrowptr+=ncols,gyrowptr+=ncols,gxrowptr+=ncols,
            resultrowptr+=ncols) {
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
        }
}
//<------------------------- end hysteresis.c ------------------------->
