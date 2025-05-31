#ifndef PARSER_H
#define PARSER_H
#include "darknet.h"
#include "network.h"
extern int total_params;
extern int partition_point1;
extern int partition_point2;
extern int frozen_bool;
extern int sepa_save_bool;
extern int count_global;
extern int global_dp;
extern int* tee_n_layer;
extern int max_tee_n_layer;
extern int use_tee_relu;
extern int run_demo_idx;
extern int current_layer_n;
extern int run_in_tee;
extern clock_t tee_total_us;
extern clock_t invoke_time_us;
extern clock_t total_run_us;
extern int cnn_layer_n;
extern int run_layer_idx;
extern int run_predict_time;
// #define RUN_FUSION
extern int vggbn_run_tee_num;
extern int resnet18_run_tee_num;
extern int resnet50_run_tee_num;
extern int mobilenetv2_run_tee_num;
// extern int mobilenetv2_run_tee[24];
extern int mobilenetv2_run_tee[25];

#ifdef RUN_FUSION
extern int resnet18_run_tee[3];
extern int vggbn_run_tee[1];
// extern int vggbn_run_tee[7];
// extern int mobilenetv2_run_tee[3];
extern int resnet50_run_tee[4];
#else
// extern int resnet18_run_tee[15];
extern int resnet18_run_tee[13];
extern int vggbn_run_tee[21];
extern int resnet50_run_tee[14];

#endif
extern int run_fusion;
// run_demo_idx:
/**
 * run_demo_idx:
 * 1 ------ resnet18
 * 2 ------ alexnet
 * 3 ------ 
 * 4 ------
 */
void save_network(network net, char *filename);
void save_weights_double(network net, char *filename);

#endif
