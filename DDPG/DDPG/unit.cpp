#include "stdafx.h"
#include "unit.h"



double Uniform(void) {
	return ((double)rand() + 1.0) / ((double)RAND_MAX + 2.0);
}

double rand_normal(double mu, double sigma) {
	double z = sqrt(-2.0*log(Uniform())) * sin(2.0*M_PI*Uniform());
	return mu + sigma * z;
} 

double *ramdom(double *x, int row, int col) {
	srand(time(NULL));
	int i;
	for (i = 0; i < row * col; i++) {
		double k;
		x[i] = Uniform();//rand_normal(0,2/sqrt((double)col));
	}
	return x;
}

network init_net(int len, int batch, int *sizes) {
	static int id = 0;
	network net;
	net.net3 = (param*)malloc(sizeof(param) * len);
	net.len = len;
	net.size = sizes;
	net.batch = batch;
	net.id = id++;
	int i;
	for (i = 0; i < len; i++) {
		net.net3[i].col = sizes[i];
		net.net3[i].row = sizes[i + 1];
		net.net3[i].w = (double*)malloc(sizeof(double) * sizes[i] * sizes[i + 1]);
		net.net3[i].wv = (double*)calloc(sizes[i] * sizes[i + 1], sizeof(double));
		net.net3[i].wm = (double*)calloc(sizes[i] * sizes[i + 1], sizeof(double));
		ramdom(net.net3[i].w, sizes[i + 1], sizes[i]);
		net.net3[i].b = (double*)calloc(sizes[i + 1], sizeof(double));
		net.net3[i].bv = (double*)calloc(sizes[i + 1], sizeof(double));
		net.net3[i].bm = (double*)calloc(sizes[i + 1], sizeof(double));
		net.net3[i].db = (double*)calloc(sizes[i + 1], sizeof(double));
		net.net3[i].dw = (double*)calloc(sizes[i] * sizes[i + 1], sizeof(double));
		net.net3[i].dx = (double*)calloc(sizes[i] * batch, sizeof(double));
		net.net3[i].y = (double*)calloc(net.size[i + 1] * batch, sizeof(double));
		net.net3[i].a_y = (double*)calloc(net.size[i + 1] * batch, sizeof(double));
	}
	return net;
}

double *linear(double *in, double *weight, double *bias, double *out, int batch, int row, int col)
{
	int i, j, k;
	for (k = 0; k < batch; k++) {
		for (i = 0; i < row; i++) {
			double sum = 0;
			for (j = 0; j < col; j++) {
				sum += in[k * col + j] * weight[j * row + i];
			}
			out[k * row + i] = sum + bias[i];
		}
	}
	return out;
}

