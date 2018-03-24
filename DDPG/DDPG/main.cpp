#include "stdafx.h"
#include "unit.h"

int main()
{	
	double total_reward = 0;
	double reward = 0;
	int state_size = 1;
	int action_size = 1;
	int q_sizes[] = {state_size+action_size,8,8,1};
	int p_sizes[] = {state_size,8,8,action_size};
	int q_len = 3;
	int batch = 100;
	int reset = 0;
	network qnet = init_net(q_len,batch,q_sizes);
	network pnet = init_net(q_len,batch,p_sizes);
	ex r = init_ex(state_size, action_size, 10000);
	double *de = (double*)calloc(qnet.batch*(r.a_size), sizeof(double));
	states s = init_state(state_size,action_size);
	int kai = 1000;
	updatestate(&s,0);
	for (int j = 0; j < 10;j++) { 
		for (int k = 0; k < kai; k++) {
			for (int i = 0; i < s.size; i++) {
				s.seq[i] = s.n_seq[i];
			}
			get_action(&pnet, s, s.seq);
			updatestate(&s,reset);
			reset = 0;
			reward = (5 - fabs(s.n_seq[0]))/10.0;
			if (reward < 0) {
				reward = -1;

			}
			total_reward += reward;
			
			add_ex(&r, s.seq, reward, s.action, s.n_seq);
			if ((5 - fabs(s.n_seq[0])) < 0) {
				s.seq[0] = 0;
				s.n_seq[0] = 0;
				reset = 1;
			}
			update_p_and_q(&pnet, &qnet, r);
			
		}printf("mean_reward:%f\n", total_reward/kai);
		total_reward = 0;
		printf("qnet_loss:%f\n", qnet.loss);
	}
	Pnetwork_predict(s.seq, &pnet);
	printf("raw_y[ %f]\n", pnet.net3[pnet.len - 1].y[0]);
	
	for (int i = r.memmaxsize-1000; i < r.memmaxsize; i++) {
		printf("seq[");
		for (int j = 0; j < r.s_size; j++) {
			printf("%f ",r.mem[i].new_state[j]);
		}printf("]");
		printf("p_seq[");
		for (int j = 0; j < r.s_size; j++) {
			printf("%f ", r.mem[i].p_state[j]);
		}printf("]");
		printf("action[");
		for (int j = 0; j < r.a_size; j++) {
			printf("%f ", r.mem[i].action[j]);
		}printf("]");
		printf("reward[%f]\n", r.mem[i].reward);
	}
	
	
    return 0;
}

