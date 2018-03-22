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

double *linear_back_bias(double *back_bias, double *back_out, int batch, int row)
{
	int i, j;
	for (i = 0; i < row; i++) {
		double sum = 0;
		for (j = 0; j < batch; j++) {

			sum += back_out[j*row + i];
		}
		back_bias[i] = sum;
	}
	return back_bias;
}

double *linear_back_weight(double *back_weight, double *back_out, double *in, int batch, int row, int col)
{
	int i, j, k;
	for (i = 0; i < col; i++) {
		for (k = 0; k < row; k++) {
			double sum = 0;
			for (j = 0; j < batch; j++) {
				sum += in[j*col + i] * back_out[j*row + k];
			}
			back_weight[i*row + k] = sum;
		}
	}
	return back_weight;
}

double *linear_back_in(double *back_in, double *back_out, double *weight, int batch, int row, int col)
{
	int i, j, k;
	for (i = 0; i < batch; i++) {
		for (k = 0; k < col; k++) {
			double sum = 0;
			for (j = 0; j < row; j++) {
				sum += back_out[i*row + j] * weight[k*row + j];
			}
			back_in[i*col + k] = sum;
		}
	}
	return back_in;
}

void linear_back(double *in, double *weight, double *bias,
	double *back_out, double *back_in, double *back_weight, double *back_bias,
	int batch, int row, int col)
{
	linear_back_bias(back_bias, back_out, batch, row);
	linear_back_weight(back_weight, back_out, in, batch, row, col);
	linear_back_in(back_in, back_out, weight, batch, row, col);
}

double *leaky_relu(double *in, double *out, int batch, int row)
{
	int i, j;
	for (i = 0; i < batch; i++) {
		for (j = 0; j < row; j++) {
			out[i * row + j] = in[i * row + j] >= 0 ?
				in[i * row + j] :
				in[i * row + j] * 0.2;
		}
	}
	return out;
}

double *leaky_relu_back(double *in, double *out, int batch, int row)
{
	int i, j;
	for (i = 0; i < batch; i++) {
		for (j = 0; j < row; j++) {
			out[i * row + j] = in[i * row + j] >= 0 ?
				out[i * row + j] :
				out[i * row + j] * 0.2;
		}
	}
	return out;
}

double *sigmoid(double *in, double *out, int batch, int row)
{
	int i, j;
	for (i = 0; i < batch; i++) {
		for (j = 0; j < row; j++) {
			out[i * row + j] = 1 / (1 + exp(-in[i * row + j]));
		}
	}
	return out;
}

double *sigmoid_back(double *out, double *dout, int batch, int row)
{
	int i, j;
	for (i = 0; i < batch; i++) {
		for (j = 0; j < row; j++) {
			dout[i * row + j] = dout[i * row + j] * out[i * row + j] * (1 - out[i * row + j]);
		}
	}
	return dout;
}

double *tanh_(double *in, double *out, int batch, int row)
{
	int i, j;
	for (i = 0; i < batch; i++) {
		for (j = 0; j < row; j++) {
			out[i * row + j] = tanh(-in[i * row + j]);
		}
	}
	return out;
}

double *tanh_back(double *out, double *dout, int batch, int row)
{
	int i, j;
	for (i = 0; i < batch; i++) {
		for (j = 0; j < row; j++) {
			dout[i * row + j] = dout[i * row + j] * out[i * row + j] * (1 - pow(out[i * row + j], 2));
		}
	}
	return dout;
}

double mean_squared_error(double *y, double *t, int batch, int row) {

	int i, j;
	double loss = 0;
	for (i = 0; i < batch; i++) {
		for (j = 0; j < row; j++) {
			loss += pow((y[i*row + j] - t[i*row + j]), 2);
		}
	}
	return 0.5*loss / batch;
}

double *mean_squared_error_back(double *y, double *t, double *back_out, int batch, int row) {
	int i, j;
	for (i = 0; i < batch; i++) {
		for (j = 0; j < row; j++) {
			back_out[i*row + j] = (y[i*row + j] - t[i*row + j]);
		}
	}
	return back_out;
}

double huber_loss(double *y, double *t, int batch, int row) {
	int i, j;
	double loss = 0;
	for (i = 0; i < batch; i++) {
		for (j = 0; j < row; j++) {
			//printf("y:%f t:%f\n", y[i*row + j], t[i*row + j]);

			double err = y[i*row + j] - t[i*row + j];
			if (fabs(err) < 1.0)
				loss += 0.5*pow(err, 2);
			else
				loss += fabs(err);
		}
	}
	return loss / batch;

}

double *huber_loss_back(double *y, double *t, double *back_out, int batch, int row) {

	int i, j;
	for (i = 0; i < batch; i++) {
		for (j = 0; j < row; j++) {
			double err = y[i*row + j] - t[i*row + j];
			if (fabs(err) < 1)
				back_out[i*row + j] = err;
			else
				back_out[i*row + j] = err > 0 ? 1 : -1;
		}
	}
	return back_out;
}


void sgd_update(double *weight, double *back_weight, double *bias, double *back_bias, double lr, int row, int col) {
	int i, j;
	for (i = 0; i < col; i++) {
		for (j = 0; j < row; j++) {
			weight[i*row + j] -= lr * back_weight[i*row + j];
		}
	}
	for (j = 0; j < row; j++) {
		bias[j] -= lr * back_bias[j];
	}

}

void adam_update(network *net) {
	net->iter++;
	double lr_t = net->lr * sqrt(1.0 - pow(net->beta2, net->iter)) / (1.0 - pow(net->beta1, net->iter));

	int i, j, k;

	for (k = 0; k < net->len; k++) {
		for (i = 0; i < net->size[k]; i++) {
			for (j = 0; j < net->size[k + 1]; j++) {
				net->net3[k].wm[i* net->size[k + 1] + j] += (1 - net->beta1) * (net->net3[k].dw[i* net->size[k + 1] + j] - net->net3[k].wm[i* net->size[k + 1] + j]);
				net->net3[k].wv[i* net->size[k + 1] + j] += (1 - net->beta2) * (pow(net->net3[k].dw[i* net->size[k + 1] + j], 2) - net->net3[k].wv[i* net->size[k + 1] + j]);
				net->net3[k].w[i* net->size[k + 1] + j] -= lr_t * net->net3[k].wm[i* net->size[k + 1] + j] / (sqrt(net->net3[k].wv[i* net->size[k + 1] + j]) + 1e-7);
			}
		}
		for (j = 0; j < net->size[k + 1]; j++) {
			net->net3[k].bm[j] += (1 - net->beta1) * (net->net3[k].db[j] - net->net3[k].bm[j]);
			net->net3[k].bv[j] += (1 - net->beta2) * (pow(net->net3[k].db[j], 2) - net->net3[k].bv[j]);
			net->net3[k].b[j] -= lr_t * net->net3[k].bm[j] / (sqrt(net->net3[k].bv[j]) + 1e-7);
		}

	}
}

void Qnetwork_predict(double *x, network *net) {
	static int batch[10] = {};

	if (batch[net->id] != net->batch) {
		batch[net->id] = net->batch;
		for (int i = 0; i < net->len; i++) {
			net->net3[i].y = (double*)calloc(net->size[i + 1] * net->batch, sizeof(double));
			net->net3[i].a_y = (double*)calloc(net->size[i + 1] * net->batch, sizeof(double));
		}
	}
	for (int i = 0; i < net->len; i++) {
		if (i == 0) {
			linear(x, net->net3[i].w, net->net3[i].b, net->net3[i].y, net->batch, net->size[i + 1], net->size[i]);
		}
		else
			linear(net->net3[i - 1].a_y, net->net3[i].w, net->net3[i].b, net->net3[i].y, net->batch, net->size[i + 1], net->size[i]);
		if (i != net->len - 1)
			leaky_relu(net->net3[i].y, net->net3[i].a_y, net->batch, net->size[i + 1]);
	}
}

void Pnetwork_predict(double *x, network *net) {
	static int batch[10] = {};

	if (batch[net->id] != net->batch) {
		batch[net->id] = net->batch;
		for (int i = 0; i < net->len; i++) {
			net->net3[i].y = (double*)calloc(net->size[i + 1] * net->batch, sizeof(double));
			net->net3[i].a_y = (double*)calloc(net->size[i + 1] * net->batch, sizeof(double));
		}
	}
	for (int i = 0; i < net->len; i++) {
		if (i == 0) {
			linear(x, net->net3[i].w, net->net3[i].b, net->net3[i].y, net->batch, net->size[i + 1], net->size[i]);
		}
		else
			linear(net->net3[i - 1].a_y, net->net3[i].w, net->net3[i].b, net->net3[i].y, net->batch, net->size[i + 1], net->size[i]);
		if (i != net->len - 1)
			leaky_relu(net->net3[i].y, net->net3[i].a_y, net->batch, net->size[i + 1]);
		else
			tanh_(net->net3[i].y, net->net3[i].y, net->batch, net->size[i + 1]);
	}
}

