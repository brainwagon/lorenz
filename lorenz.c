#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define sigma	10.
#define beta	8./3.
#define rho	28.

void
lorenz(double P[3], double dP[3])
{
    dP[0] = sigma * (P[1] - P[0]) ;
    dP[1] = P[0] * (rho - P[2]) - P[1] ;
    dP[2] = P[0] * P[1] - beta * P[2] ;
}

void
euler(double P[3], double h, double nP[3])
{
    double dP[3] ;

    lorenz(P, dP) ;

    nP[0] = P[0] + h * dP[0] ; 
    nP[1] = P[1] + h * dP[1] ; 
    nP[2] = P[2] + h * dP[2] ; 
}

void
rk4(double P[3], double h, double nP[3])
{
    double k[4][3] ;
    double tmp[3] ;
    lorenz(P, k[0]) ;
    tmp[0] = P[0] + 0.5 * h * k[0][0] ;
    tmp[1] = P[1] + 0.5 * h * k[0][1] ;
    tmp[2] = P[2] + 0.5 * h * k[0][2] ;
    lorenz(tmp, k[1]) ;
    tmp[0] = P[0] + 0.5 * h * k[1][0] ;
    tmp[1] = P[1] + 0.5 * h * k[1][1] ;
    tmp[2] = P[2] + 0.5 * h * k[1][2] ;
    lorenz(tmp, k[2]) ;
    tmp[0] = P[0] +       h * k[2][0] ;
    tmp[1] = P[1] +       h * k[2][1] ;
    tmp[2] = P[2] +       h * k[2][2] ;
    lorenz(tmp, k[3]) ;

    nP[0] = P[0] + h / 6. * (k[0][0] + 2.0*k[1][0] + 2.0*k[2][0] + k[3][0]) ;
    nP[1] = P[1] + h / 6. * (k[0][1] + 2.0*k[1][1] + 2.0*k[2][1] + k[3][1]) ;
    nP[2] = P[2] + h / 6. * (k[0][2] + 2.0*k[1][2] + 2.0*k[2][2] + k[3][2]) ;
}


#define XSIZE	1280
#define YSIZE	720

int imgR[YSIZE][XSIZE] ;
int imgG[YSIZE][XSIZE] ;

double 
grand()
{
    int i ;
    double t = 0. ;

    for (i=0; i<16; i++) 
	t += drand48() - 0.5 ;
    return t ;
}

void
plot(int img[YSIZE][XSIZE], double x, double y)
{
    int ix, iy ;
    int p ;

    x = ((x + 20) / 40.) * XSIZE ;
    y = (y / 50.) * YSIZE ;

    for (p=1; p<100; p++) {
	ix = round(x + grand()) ;
	iy = round(y + grand()) ;
	if (ix >= 0 && ix < XSIZE && iy >= 0 && iy < YSIZE)
	    img[iy][ix] ++ ;
    }
}

void
doframe(double P[3], double Q[3], double h, double l, int frame)
{
    double t ;
    int x, y ;
    int m = 0 ;
    char cmd[80] ;

    fprintf(stderr, "... doing frame %d\n", frame) ;
    for (y=0; y<YSIZE; y++) 
	for (x=0; x<XSIZE; x++) {
	    imgR[y][x] *= 0.9 ;
	    imgG[y][x] *= 0.9 ;
        }

    fprintf(stderr, "... marching particles\n") ;
    for (t=0; t<l; t+=h) {
	rk4(P, h, P) ;
	plot(imgR, P[0], P[2]) ;
	rk4(Q, h, Q) ;
	plot(imgG, Q[0], Q[2]) ;
    }

    fprintf(stderr, "... normalizing particles\n") ;
    for (y=0; y<YSIZE; y++) 
	for (x=0; x<XSIZE; x++) {
	    if (imgR[y][x] > m)
	    	m = imgR[y][x] ;
	    if (imgG[y][x] > m)
	    	m = imgG[y][x] ;
        }

    fprintf(stderr, "m = %d\n", m) ;

    sprintf(cmd, "lorenz.%04d.ppm", frame+1) ;
    FILE *fp = fopen(cmd, "w") ;
    fprintf(fp, "P6\n%d %d\n255\n", XSIZE, YSIZE) ;
    for (y=0; y<YSIZE; y++) 
	for (x=0; x<XSIZE; x++) {
	    double t = (double) imgR[y][x] / m ;
	    if (t < 0) t = 0.0 ;
	    if (t > 1) t = 1.0 ;
	    fputc((int) (255.*pow(t, 0.4545)), fp) ;
	    t = (double) imgG[y][x] / m ;
	    if (t < 0) t = 0.0 ;
	    if (t > 1) t = 1.0 ;
	    fputc((int) (255.*pow(t, 0.4545)), fp) ;
            fputc(0, fp) ;
	}

    fclose(fp) ;
}

int 
main(int argc, char *argv[])
{
    int i, x, y ;
    double P[3], Q[3] ;
    double h = 0.0001 ;
    double t = 0. ;
    double l = .05 ;
    int frame ;

    P[0] = 1. ;
    P[1] = 1. ;
    P[2] = 1.0000001 ;

    Q[0] = 1. ;
    Q[1] = 1. ;
    Q[2] = 1. ;

    for (y=0; y<YSIZE; y++) 
	for (x=0; x<XSIZE; x++) 
	    imgR[y][x] = imgG[y][x] = 0. ;

    for (i=0; i<1000; i++)
	rk4(P, 0.1, P) ;
    Q[0] = P[0] ;
    Q[1] = P[1] ;
    Q[2] = P[2] + 1e-4 ;

    for (frame=0; frame<1200; frame++) 
	doframe(P, Q, h, l, frame) ;

    /* normalize */

}
