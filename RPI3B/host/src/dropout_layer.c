#include "dropout_layer.h"
#include "utils.h"
#include "opencl.h"
#include "parser.h"
#include <stdlib.h>
#include <stdio.h>

dropout_layer make_dropout_layer(int batch, int inputs, float probability)
{
    dropout_layer l = {0};
    l.type = DROPOUT;
    l.probability = probability;
    l.inputs = inputs;
    l.outputs = inputs;
    l.batch = batch;
    l.rand = calloc(inputs*batch, sizeof(float));
    l.scale = 1./(1.-probability);
    l.forward = forward_dropout_layer;
    l.backward = backward_dropout_layer;
#ifdef GPU
    if (gpu_index >= 0) {
        l.forward_gpu = forward_dropout_layer_gpu;
        l.backward_gpu = backward_dropout_layer_gpu;
        l.update_gpu = 0;
        l.rand_gpu = opencl_make_array(l.rand, inputs * batch);
    }
#endif

    // if(count_global <= partition_point1 || count_global > partition_point2){
    if(run_in_tee == 0){
        fprintf(stderr, "dropout       p = %.2f               %4d  ->  %4d\n", probability, inputs, inputs);
    }else{
        fprintf(stderr, "dropout_TA    p = %.2f               %4d  ->  %4d\n", probability, inputs, inputs);
    }
    return l;
}

void resize_dropout_layer(dropout_layer *l, int inputs)
{
#ifdef GPU
    if (gpu_index >= 0) {
        opencl_free_gpu_only(l->rand_gpu);
    }
#endif
    l->rand = realloc(l->rand, l->inputs*l->batch*sizeof(float));
#ifdef GPU
    if (gpu_index >= 0) {
        l->rand_gpu = opencl_make_array(l->rand, inputs * l->batch);
    }
#endif
}

void forward_dropout_layer(dropout_layer l, network net)
{
    int i;
    if (!net.train) return;
    for(i = 0; i < l.batch * l.inputs; ++i){
        //printf("i = %d; total = %d\n",i, l.batch * l.inputs);
        float r = rand_uniform(0, 1);
        l.rand[i] = r;
        if(r < l.probability) net.input[i] = 0;
        else net.input[i] *= l.scale;
    }
}

void backward_dropout_layer(dropout_layer l, network net)
{
    int i;
    if(!net.delta) return;
    for(i = 0; i < l.batch * l.inputs; ++i){
        float r = l.rand[i];
        if(r < l.probability) net.delta[i] = 0;
        else net.delta[i] *= l.scale;
    }
}
