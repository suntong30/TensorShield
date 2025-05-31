#ifndef CONNECTED_LAYER_TA_H
#define CONNECTED_LAYER_TA_H
#include "darknet_TA.h"

extern int MIN_INPUTS;
void forward_connected_layer_TA_new(layer_TA l, network_TA net);
void forward_connected_layer_TA_new_part(layer_TA l, network_TA net);
void backward_connected_layer_TA_new(layer_TA l, network_TA net);

void update_connected_layer_TA_new(layer_TA l, update_args_TA a);
void forward_init(layer_TA l, network_TA net);
void forward_gemm_part(layer_TA l, network_TA net);
void forward_after(layer_TA l, network_TA net);
layer_TA make_connected_layer_TA_new(int batch, int inputs, int outputs, ACTIVATION_TA activation, int batch_normalize, int adam);

#endif
