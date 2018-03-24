#include "stdafx.h"
#include "math.h"
#include <time.h>
#include <stdlib.h>
#include "string.h"

#define M_PI 3.14159


typedef struct {
	double *p_state;
	double reward = 0;
	double *action;
	double *new_state;
}exmem;

typedef struct {
	double *p_state;
	double *reward;
	double *action;
	double *new_state;
}exbatch;

typedef struct {
	int memmaxsize;
	int pos = 0;
	int size = 0;
	int s_size = 0;
	int a_size = 0;
	exmem *mem;
}ex;

typedef struct {
	double *state;
	double *action;
	double *n_seq;
	double *seq;
	int size;
}states;

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

double *linear(double *in, double *weight, double *bias, double *out, int batch, int row, int col);
double *linear_back_bias(double *back_bias, double *back_out, int batch, int row);
double *linear_back_weight(double *back_weight, double *back_out, double *in, int batch, int row, int col);
double *linear_back_in(double *back_in, double *back_out, double *weight, int batch, int row, int col);
void linear_back(double *in, double *weight, double *bias,
	double *back_out, double *back_in, double *back_weight, double *back_bias,
	int batch, int row, int col);

double *leaky_relu(double *in, double *out, int batch, int row);
double *leaky_relu_back(double *in, double *out, int batch, int row);
double *sigmoid(double *in, double *out, int batch, int row);
double *sigmoid_back(double *out, double *dout, int batch, int row);
double *tanh_(double *in, double *out, int batch, int row);
double *tanh_back(double *out, double *dout, int batch, int row);

double mean_squared_error(double *y, double *t, int batch, int row);
double *mean_squared_error_back(double *y, double *t, double *back_out, int batch, int row);
double huber_loss(double *y, double *t, int batch, int row);
double *huber_loss_back(double *y, double *t, double *back_out, int batch, int row);

void sgd_update(double *weight, double *back_weight, double *bias, double *back_bias, double lr, int row, int col);
void adam_update(network *net);

void Qnetwork_predict(double *x, network *net);
void Pnetwork_predict(double *x, network *net);

void Qnetwork_train(double *x, double *targetsQ, network *net);
void Pnetwork_train(double *x, double *de, network *net);

states init_state(int size, int size_a);
ex init_ex(int sizeof_s, int sizeof_a, int max_size);
void add_ex(ex *m, double *p_s, double reward, double *action, double *n_s);

void updatestate(states *state, int reset);
void get_action(network *p, states state, double *seq);

void random_data(network *p, ex e, exbatch *data);

void update_p_and_q(network *p, network *q, ex e);
