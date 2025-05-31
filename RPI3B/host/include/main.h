#ifndef MAIN_CA_H
#define MAIN_CA_H

#include <err.h>
#include <stdio.h>
#include <string.h>
#include "parser.h"
/* OP-TEE TEE client API (built by optee_client) */
#include <tee_client_api.h>
#include "darknet.h"
extern int current_layer_n;
#define MAKE_NETWORK_CMD 1
#define WORKSPACE_NETWORK_CMD 2
#define MAKE_CONV_CMD 3
#define MAKE_MAX_CMD 4
#define MAKE_DROP_CMD 5
#define MAKE_CONNECTED_CMD 6
#define MAKE_SOFTMAX_CMD 7
#define MAKE_COST_CMD 8
#define FORWARD_CMD 9
#define BACKWARD_CMD 10
#define BACKWARD_ADD_CMD 11
#define UPDATE_CMD 12
#define NET_TRUTH_CMD 13
#define CALC_LOSS_CMD 14
#define TRANS_WEI_CMD 15
#define OUTPUT_RETURN_CMD 16
#define SAVE_WEI_CMD 17

#define FORWARD_BACK_CMD 18
#define BACKWARD_BACK_CMD 19
#define BACKWARD_BACK_ADD_CMD 20

#define MAKE_AVG_CMD 21
#define FORWARD_CMD_TEST 22
#define FORWARD_BACK_CMD_ONE_LAYER 26

#define FUSION_TEST 23
#define MAKE_ACTIVATE_CMD 24
#define FORWARD_CMD_NO_CONV_TEST 25
#define FORWARD_CMD_RELU 27
#define INIT_CMD_MASK 28
#define FORWARD_CMD_PER_LAYER 29
#define FREE_CMD 30
#define FORWARD_CMD_PART 31
#define MAKE_SHORTCUT_LAYER_CMD 32
#define FIRST_N_CMD 33
#define FREE_N_CMD 34
#define RETURN_RUN_TIME_CMD 35
#define FORWARD_CMD_CONNECT_LAYER_PART 36
#define MAKE_BATCHNORM_CMD 37

#define TA_DARKNETP_UUID \
	{ 0x7fc5c039, 0x0542, 0x4ee1, \
		{ 0x80, 0xaf, 0xb4, 0xea, 0xb2, 0xf1, 0x99, 0x8d} }

extern TEEC_Context ctx;
extern TEEC_Session sess;

extern float *net_input_back;
extern float *net_delta_back;
extern float *net_output_back;
extern char state;

void debug_plot(char *filename, int num, float *tobeplot, int length);

void make_network_CA(int n, float learning_rate, float momentum, float decay, int time_steps, int notruth, int batch, int subdivisions, int random, int adam, float B1, float B2, float eps, int h, int w, int c, int inputs, int max_crop, int min_crop, float max_ratio, float min_ratio, int center, float clip, float angle, float aspect, float saturation, float exposure, float hue, int burn_in, float power, int max_batches);

void update_net_agrv_CA_allocateSM(int workspace_size, float *workspace);

void update_net_agrv_CA(int cond, int workspace_size, float *workspace);

void make_convolutional_layer_CA(int batch, int h, int w, int c, int n, int groups, int size, int stride, int padding, ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam, int flipped, float dot);

void make_maxpool_layer_CA(int batch, int h, int w, int c, int size, int stride, int padding);

void make_avgpool_layer_CA(int batch, int h, int w, int c);

void make_dropout_layer_CA(int batch, int inputs, float probability, int w, int h, int c, float *net_prev_output, float *net_prev_delta);

void make_connected_layer_CA(int batch, int inputs, int outputs, ACTIVATION activation, int batch_normalize, int adam);

void make_softmax_layer_CA(int batch, int inputs, int groups, float temperature, int w, int h, int c, int spatial, int noloss);

void make_cost_layer_CA(int batch, int inputs, COST_TYPE cost_type, float scale, float ratio, float noobject_scale, float thresh);

void make_shortcut_layer_CA(int batch, int index, int w, int h, int c, int from_out_w, int from_out_h, int from_out_c);
void forward_connect_layer_part_CA(layer_info_and_weights layer_info, float *input, int input_size, float *weights, float *biases, 
    float *scales, float *rolling_mean, float *variance);

void forward_network_CA(float *net_input, int net_inputs, int net_batch, int net_train);
void forward_network_CA_test(float *net_input, int size);
void forward_network_CA_fusion(float *net_input, int size, int channel);
void tt_forward_network_back_one_layer_CA(float *l_output, int net_inputs, int net_batch, int n_layer_th);
void forward_network_CA_one_layer(float *net_input, int l_inputs, int net_batch, int net_train, int n_layer_th);
void make_activation_layer_CA(int batch, int inputs, ACTIVATION activation);
void make_batchnorm_layer_CA(int batch, int h, int w, int c);
// void forward_network_CA_relu(float *output, int size, int activation, int out_channels, int size_y, int use_mask);
void init_mask(void);
void get_tee_run_time(uint32_t *deobj_run_ms, uint32_t *mask_run_ms, uint32_t *tee_run_ms);
void forward_network_CA_relu(float *output, int size, int activation, int out_channels, int size_y, int use_mask, int flops);
// void forward_network_CA_relu(float *output, int size, int activation);
void forward_network_back_CA(float *l_output, int net_inputs, int net_batch);

void backward_network_CA_addidion(float *l_output, float *l_delta, int net_inputs, int net_batch);

void backward_network_CA(float *net_input, int l_inputs, int batch, int net_train);

void backward_network_back_CA_addidion(float *l_output, float *l_delta, int net_inputs, int net_batch);

void backward_network_back_CA(float *net_input, int l_inputs, int net_batch, float *net_delta);

void update_network_CA(update_args a);

void net_truth_CA(float *net_truth, int net_truths, int net_batch);

void calc_network_loss_CA(int n, int batch);

void net_output_return_CA(int net_outputs, int net_batch);

void transfer_weights_CA(float *vec, int length, int layer_i, char type, int additional);

void save_weights_CA(float *vec, int length, int layer_i, char type);

void summary_array(char *print_name, float *arr, int n);
// void forward_network_per_layer_CA(layer_info_and_weights layer_info, float *input, int input_size, float *weights, float *biases);
void forward_network_per_layer_CA(layer_info_and_weights layer_info, float *input, int input_size, float *weights, float *biases, 
    float *scales, float *rolling_mean, float *variance);
void free_mask(void);
// void forward_network_part_CA(float *net_input, int l_inputs, int net_batch, int net_train, int start_idx, int *next_idx);
// void forward_network_part_CA(float *net_input, int l_inputs, int net_batch, int net_train, int start_idx, int end_idx);
void forward_network_part_CA(float *net_input, int l_inputs, int net_batch, int net_train, int start_idx, int end_idx, float *output, int outputs);
void first_forward_network_part_CA(void);
void free_forward_network_part_CA(void);
#endif

