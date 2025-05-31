#ifndef NORMALIZATION_LAYER_TA_H
#define NORMALIZATION_LAYER_TA_H
#include "darknet_TA.h"

layer_TA make_normalization_layer_TA(int batch, int w, int h, int c, int size, float alpha, float beta, float kappa);
void resize_normalization_layer_TA(layer_TA *layer, int h, int w);
void forward_normalization_layer_TA(const layer_TA layer, network_TA net);
void backward_normalization_layer_TA(const layer_TA layer, network_TA net);
#endif