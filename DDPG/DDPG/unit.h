#include "stdafx.h"
#include "math.h"
#include <time.h>
#include <stdlib.h>
#include "string.h"

#define M_PI 3.14159


typedef struct {
	int col;
	int row;
	double *w;
	double *b;
	double *y;
	double *a_y;
	double *dw;
	double *dx;
	double *db;
	double *wv;
	double *wm;
	double *bv;
	double *bm;

}param;

typedef struct {
	param *net3;
	double *de;
	double lr = 0.001;
	double beta1 = 0.9;
	double beta2 = 0.999;
	int iter = 0;
	double loss = 0;
	int len;
	int batch;
	int *size;
	int id;

}network;

double Uniform(void);
double rand_normal(double mu, double sigma);
double *ramdom(double *x, int row, int col);

network init_net(int len, int batch, int *sizes);

