#ifndef SHORTCUT_layer_TA_H
#define SHORTCUT_layer_TA_H

#include "darknet_TA.h"
layer_TA make_shortcut_layer_TA(int batch, int index, int w, int h, int c, int w2, int h2, int c2);
void forward_shortcut_layer_TA(const layer_TA l, network_TA net);
void backward_shortcut_layer_TA(const layer_TA l, network_TA net);
void resize_shortcut_layer_TA(layer_TA *l, int w, int h);


#endif
