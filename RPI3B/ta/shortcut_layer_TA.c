#include "shortcut_layer_TA.h"
#include "darknet_TA.h"
#include "activations_TA.h"
#include "blas_TA.h"
#include "parser_TA.h"

layer_TA make_shortcut_layer_TA(int batch, int index, int w, int h, int c, int w2, int h2, int c2)
{
    layer_TA l = {0};
    l.type = SHORTCUT_TA;
    l.batch = batch;
    l.w = w2;
    l.h = h2;
    l.c = c2;
    l.out_w = w;
    l.out_h = h;
    l.out_c = c;
    l.outputs = w*h*c;
    l.inputs = l.outputs;
    l.add = calloc(w2*h2*c2*l.batch, sizeof(float));
    if(l.add == NULL)
    {
        printf("[SHORTCUT] l.add failed to calloc\n");
    }
    l.index = index;
    l.output = calloc(l.outputs*batch, sizeof(float));

    if(l.output == NULL)
    {
        printf("[SHORTCUT] l.output failed to calloc\n");
    }
    l.forward_TA = forward_shortcut_layer_TA;
#ifdef TRAIN_NET
    l.backward_TA = backward_shortcut_layer_TA;
    l.delta =  calloc(l.outputs*batch, sizeof(float));
#endif
    return l;
}

void forward_shortcut_layer_TA(const layer_TA l, network_TA net)
{
    copy_cpu_TA(l.outputs*l.batch, net.input, 1, l.output, 1);
    shortcut_cpu_TA(l.batch, l.w, l.h, l.c, l.add, l.out_w, l.out_h, l.out_c, l.alpha, l.beta, l.output);
    activate_array_TA(l.output, l.outputs*l.batch, l.activation);
}
