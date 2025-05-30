#include "shortcut_layer.h"
#include "opencl.h"
#include "blas.h"
#include "activations.h"
#include "parser.h"
#include <stdio.h>
#include <assert.h>

layer make_shortcut_layer(int batch, int index, int w, int h, int c, int w2, int h2, int c2)
{
    if(run_in_tee == 0)
    {
        fprintf(stderr, "res  %3d                %4d x%4d x%4d   ->  %4d x%4d x%4d\n",index, w2,h2,c2, w,h,c);
    }
    else 
    {
        fprintf(stderr, "res TA[unfinished]  %3d                %4d x%4d x%4d   ->  %4d x%4d x%4d\n",index, w2,h2,c2, w,h,c);
    }
    layer l = {0};
    l.type = SHORTCUT;
    l.batch = batch;
    l.w = w2;
    l.h = h2;
    l.c = c2;
    l.out_w = w;
    l.out_h = h;
    l.out_c = c;
    l.outputs = w*h*c;
    l.inputs = l.outputs;

    l.index = index;

    l.delta =  calloc(l.outputs*batch, sizeof(float));
    l.output = calloc(l.outputs*batch, sizeof(float));;

    l.forward = forward_shortcut_layer;
    l.backward = backward_shortcut_layer;
#ifdef GPU
    if (gpu_index >= 0) {
        l.forward_gpu = forward_shortcut_layer_gpu;
        l.backward_gpu = backward_shortcut_layer_gpu;
        l.update_gpu = 0;
        l.delta_gpu = opencl_make_array(l.delta, l.outputs*batch);
        l.output_gpu = opencl_make_array(l.output, l.outputs*batch);
    }
#endif
    return l;
}

void resize_shortcut_layer(layer *l, int w, int h)
{
    assert(l->w == l->out_w);
    assert(l->h == l->out_h);
    l->w = l->out_w = w;
    l->h = l->out_h = h;
#ifdef GPU
    if (gpu_index >= 0) {
        opencl_free_gpu_only(l->output_gpu);
        opencl_free_gpu_only(l->delta_gpu);
    }
#endif
    l->outputs = w*h*l->out_c;
    l->inputs = l->outputs;
    l->delta =  realloc(l->delta, l->outputs*l->batch*sizeof(float));
    l->output = realloc(l->output, l->outputs*l->batch*sizeof(float));

#ifdef GPU
    if (gpu_index >= 0) {
        l->output_gpu = opencl_make_array(l->output, l->outputs*l->batch);
        l->delta_gpu = opencl_make_array(l->delta, l->outputs*l->batch);
    }
#endif
    
}


void forward_shortcut_layer(const layer l, network net)
{
    copy_cpu(l.outputs*l.batch, net.input, 1, l.output, 1);
    shortcut_cpu(l.batch, l.w, l.h, l.c, net.layers[l.index].output, l.out_w, l.out_h, l.out_c, l.alpha, l.beta, l.output);
    // activate_array(l.output, l.outputs*l.batch, l.activation);
    if (use_tee_relu == 1 && (run_demo_idx == 0  || (run_demo_idx > 0 && run_in_tee > 0)))
    {
        forward_network_CA_relu(l.output,l.outputs*l.batch, l.activation, l.out_c, l.out_h, cnn_layer_n, 0);
    }
    else {
        activate_array(l.output, l.outputs*l.batch, l.activation);
    }
    cnn_layer_n++;
}

void backward_shortcut_layer(const layer l, network net)
{
    gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);
    axpy_cpu(l.outputs*l.batch, l.alpha, l.delta, 1, net.delta, 1);
    shortcut_cpu(l.batch, l.out_w, l.out_h, l.out_c, l.delta, l.w, l.h, l.c, 1, l.beta, net.layers[l.index].delta);
}

#ifdef GPU
void forward_shortcut_layer_gpu(const layer l, network net)
{
    copy_gpu(l.outputs*l.batch, net.input_gpu, 1, l.output_gpu, 1);
    shortcut_gpu(l.batch, l.w, l.h, l.c, net.layers[l.index].output_gpu, l.out_w, l.out_h, l.out_c, l.alpha, l.beta, l.output_gpu);
    // activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);
    if (use_tee_relu == 1 && (run_demo_idx == 0 || (run_demo_idx > 0 && run_in_tee > 0)))
    {
        opencl_pull_array(l.output_gpu, l.output, l.outputs*l.batch);
        forward_network_CA_relu(l.output,l.outputs*l.batch, l.activation, l.out_c, l.out_h, cnn_layer_n, 0);
        opencl_push_array(l.output_gpu, l.output, l.outputs*l.batch);
    }
    else {
        activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);
    }
    cnn_layer_n++;
}

void backward_shortcut_layer_gpu(const layer l, network net)
{
    gradient_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);
    axpy_gpu(l.outputs*l.batch, l.alpha, l.delta_gpu, 1, net.delta_gpu, 1);
    shortcut_gpu(l.batch, l.out_w, l.out_h, l.out_c, l.delta_gpu, l.w, l.h, l.c, 1, l.beta, net.layers[l.index].delta_gpu);
}
#endif
