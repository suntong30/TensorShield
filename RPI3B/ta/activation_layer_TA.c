#include "activation_layer_TA.h"
#include "darknet_TA.h"
#include "activations_TA.h"
#include <tee_internal_api.h>
#include <tee_internal_api_extensions.h>
#include "blas_TA.h"


static size_t get_workspace_size(layer_TA l){
    return (size_t)l.outputs*sizeof(float);
}
// tt_make_activation_layer
layer_TA make_activation_layer_TA(int batch, int inputs, ACTIVATION_TA activation)
{
	layer_TA l = { 0 };
    l.type = ACTIVE_TA;

    l.inputs = inputs;
    l.outputs = inputs;
    l.batch=batch;

    l.output = (float*)calloc(batch*inputs, sizeof(float*));
    if (l.output == NULL)
    {
        printf("[ACTIVATION] l.output failed to malloc\n");
    }

    l.forward_TA = forward_activation_layer_TA;
    // l.backward = backward_activation_layer;

#ifdef TRAIN_NET
    l.delta = (float*)calloc(batch*inputs, sizeof(float*));
    l.backward_TA = tt_backward_activation_layer_TA_new;
#endif
    l.activation = activation;
    return l;
}

void forward_activation_layer_TA(layer_TA l, network_TA net)
{
    copy_cpu_TA(l.outputs*l.batch, net.input, 1, l.output, 1);
    activate_array_TA(l.output, l.outputs*l.batch, l.activation);
}

void backward_activation_layer_TA(layer_TA l, network_TA net)
{
    gradient_array_TA(l.output, l.outputs*l.batch, l.activation, l.delta);
    copy_cpu_TA(l.outputs*l.batch, l.delta, 1, net.delta, 1);
}
