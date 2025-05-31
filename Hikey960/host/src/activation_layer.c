#include "activation_layer.h"
#include "utils.h"
#include "opencl.h"
#include "blas.h"
#include "gemm.h"
#include "parser.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

layer make_activation_layer(int batch, int inputs, ACTIVATION activation)
{
    layer l = {0};
    l.type = ACTIVE;

    l.inputs = inputs;
    l.outputs = inputs;
    l.batch=batch;

    l.output = calloc(batch*inputs, sizeof(float*));
    l.delta = calloc(batch*inputs, sizeof(float*));

    l.forward = forward_activation_layer;
    l.backward = backward_activation_layer;
#ifdef GPU
    if (gpu_index >= 0) {
        l.forward_gpu = forward_activation_layer_gpu;
        l.backward_gpu = backward_activation_layer_gpu;
        l.update_gpu = 0;
        l.output_gpu = opencl_make_array(l.output, inputs * batch);
        l.delta_gpu = opencl_make_array(l.delta, inputs * batch);
    }
#endif
    l.activation = activation;
    if (run_in_tee == 0)
    {
        fprintf(stderr, "Activation Layer: %d inputs\n", inputs);
    }
    else
    {
        fprintf(stderr, "Activation LayerTA[unfinished]: %d inputs\n", inputs);
    }
    return l;
}

void forward_activation_layer(layer l, network net)
{
    copy_cpu(l.outputs*l.batch, net.input, 1, l.output, 1);
    activate_array(l.output, l.outputs*l.batch, l.activation);
}

void backward_activation_layer(layer l, network net)
{
    gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);
    copy_cpu(l.outputs*l.batch, l.delta, 1, net.delta, 1);
}

#ifdef GPU

void forward_activation_layer_gpu(layer l, network net)
{
    copy_gpu(l.outputs*l.batch, net.input_gpu, 1, l.output_gpu, 1);
    if (use_tee_relu == 1 && (run_demo_idx == 0 || (run_demo_idx > 0 && run_in_tee > 0)))
    {
        opencl_pull_array(l.output_gpu, l.output, l.outputs*l.batch);
        forward_network_CA_relu(l.output,l.outputs*l.batch, l.activation, l.out_c, l.out_h, cnn_layer_n, (2.0 * l.n * l.size*l.size*l.c/l.groups * l.out_h*l.out_w));
        opencl_push_array(l.output_gpu, l.output, l.outputs*l.batch);
    }
    else {
        activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);
    }

}

void backward_activation_layer_gpu(layer l, network net)
{
    gradient_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);
    copy_gpu(l.outputs*l.batch, l.delta_gpu, 1, net.delta_gpu, 1);
}
#endif
