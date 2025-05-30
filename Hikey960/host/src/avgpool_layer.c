#include "avgpool_layer.h"
#include "opencl.h"
#include "parser.h"

#include <stdio.h>

avgpool_layer make_avgpool_layer(int batch, int w, int h, int c)
{

    // if(count_global <= partition_point1 || count_global > partition_point2){
    if(run_in_tee == 0){
        fprintf(stderr, "avg                     %4d x%4d x%4d   ->  %4d\n",  w, h, c, c);
    }
    else{
        fprintf(stderr, "avg_TA                  %4d x%4d x%4d   ->  %4d\n",  w, h, c, c);
    }
    avgpool_layer l = {0};
    l.type = AVGPOOL;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    l.out_w = 1;
    l.out_h = 1;
    l.out_c = c;
    l.outputs = l.out_c;
    l.inputs = h*w*c;
    int output_size = l.outputs * batch;
    l.output =  calloc(output_size, sizeof(float));
    l.delta =   calloc(output_size, sizeof(float));
    l.forward = forward_avgpool_layer;
    l.backward = backward_avgpool_layer;
    #ifdef GPU
    if (gpu_index >= 0) {
        l.forward_gpu = forward_avgpool_layer_gpu;
        l.backward_gpu = backward_avgpool_layer_gpu;
        l.update_gpu = 0;
        l.output_gpu = opencl_make_array(l.output, output_size);
        l.delta_gpu = opencl_make_array(l.delta, output_size);
    }
    #endif
    return l;
}

void resize_avgpool_layer(avgpool_layer *l, int w, int h)
{
    l->w = w;
    l->h = h;
    l->inputs = h*w*l->c;
}

void forward_avgpool_layer(const avgpool_layer l, network net)
{
    int b,i,k;

    for(b = 0; b < l.batch; ++b){
        for(k = 0; k < l.c; ++k){
            int out_index = k + b*l.c;
            l.output[out_index] = 0;
            for(i = 0; i < l.h*l.w; ++i){
                int in_index = i + l.h*l.w*(k + b*l.c);
                l.output[out_index] += net.input[in_index];
            }
            l.output[out_index] /= l.h*l.w;
        }
    }
}

void backward_avgpool_layer(const avgpool_layer l, network net)
{
    int b,i,k;

    for(b = 0; b < l.batch; ++b){
        for(k = 0; k < l.c; ++k){
            int out_index = k + b*l.c;
            for(i = 0; i < l.h*l.w; ++i){
                int in_index = i + l.h*l.w*(k + b*l.c);
                net.delta[in_index] += l.delta[out_index] / (l.h*l.w);
            }
        }
    }
}
