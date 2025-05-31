#ifndef ACTIVATION_LAYER_TA_H
#define ACTIVATION_LAYER_TA_H

#include "darknet_TA.h"

layer_TA make_activation_layer_TA(int batch, int inputs, ACTIVATION_TA activation);

void forward_activation_layer_TA(layer_TA l,network_TA net);
void backward_activation_layer_TA(layer_TA l, network_TA net);
#endif
