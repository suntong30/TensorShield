#include "normalization_layer_TA.h"
#include "blas_TA.h"
#include <stdio.h>
#include <tee_internal_api.h>
#include <tee_internal_api_extensions.h>

layer_TA make_normalization_layer_TA(int batch, int w, int h, int c, int size, float alpha, float beta, float kappa)
{
    layer_TA layer = {0};
    layer.type = NORMALIZATION_TA;
    layer.batch = batch;
    layer.h = layer.out_h = h;
    layer.w = layer.out_w = w;
    layer.c = layer.out_c = c;
    layer.kappa = kappa;
    layer.size = size;
    layer.alpha = alpha;
    layer.beta = beta;
    layer.output = calloc(h * w * c * batch, sizeof(float));
    layer.delta = calloc(h * w * c * batch, sizeof(float));
    layer.squared = calloc(h * w * c * batch, sizeof(float));
    layer.norms = calloc(h * w * c * batch, sizeof(float));
    if (layer.output == NULL)
    {
        printf("[NORMALIZATION] output failed to calloc\n");
    }
    if (layer.delta == NULL)
    {
        printf("[NORMALIZATION] delta failed to calloc\n");
    }
    if (layer.squared == NULL)
    {
        printf("[NORMALIZATION] squared failed to calloc\n");
    }
    if (layer.norms == NULL)
    {
        printf("[NORMALIZATION] norms failed to calloc\n");
    }
    layer.inputs = w*h*c;
    layer.outputs = layer.inputs;

    layer.forward_TA = forward_normalization_layer_TA;
    layer.backward_TA = backward_normalization_layer_TA;
    return layer;
}
void resize_normalization_layer_TA(layer_TA *layer, int h, int w)
{
    int c = layer->c;
    int batch = layer->batch;
    layer->h = h;
    layer->w = w;
    layer->out_h = h;
    layer->out_w = w;
    layer->inputs = w*h*c;
    layer->outputs = layer->inputs;
    layer->output = realloc(layer->output, h * w * c * batch * sizeof(float));
    layer->delta = realloc(layer->delta, h * w * c * batch * sizeof(float));
    layer->squared = realloc(layer->squared, h * w * c * batch * sizeof(float));
    layer->norms = realloc(layer->norms, h * w * c * batch * sizeof(float));
}
void forward_normalization_layer_TA(const layer_TA layer, network_TA net)
{
    int k,b;
    int w = layer.w;
    int h = layer.h;
    int c = layer.c;
    scal_cpu_TA(w*h*c*layer.batch, 0, layer.squared, 1);

    for(b = 0; b < layer.batch; ++b){
        float *squared = layer.squared + w*h*c*b;
        float *norms   = layer.norms + w*h*c*b;
        float *input   = net.input + w*h*c*b;
        pow_cpu_TA(w*h*c, 2, input, 1, squared, 1);

        const_cpu_TA(w*h, layer.kappa, norms, 1);
        for(k = 0; k < layer.size/2; ++k){
            axpy_cpu_TA(w*h, layer.alpha, squared + w*h*k, 1, norms, 1);
        }

        for(k = 1; k < layer.c; ++k){
            copy_cpu_TA(w*h, norms + w*h*(k-1), 1, norms + w*h*k, 1);
            int prev = k - ((layer.size-1)/2) - 1;
            int next = k + (layer.size/2);
            if(prev >= 0)      axpy_cpu_TA(w*h, -layer.alpha, squared + w*h*prev, 1, norms + w*h*k, 1);
            if(next < layer.c) axpy_cpu_TA(w*h,  layer.alpha, squared + w*h*next, 1, norms + w*h*k, 1);
        }
    }
    pow_cpu_TA(w*h*c*layer.batch, -layer.beta, layer.norms, 1, layer.output, 1);
    mul_cpu_TA(w*h*c*layer.batch, net.input, 1, layer.output, 1);

}
void backward_normalization_layer_TA(const layer_TA layer, network_TA net)
{
    int w = layer.w;
    int h = layer.h;
    int c = layer.c;
    pow_cpu_TA(w*h*c*layer.batch, -layer.beta, layer.norms, 1, net.delta, 1);
    mul_cpu_TA(w*h*c*layer.batch, layer.delta, 1, net.delta, 1);

}
void visualize_normalization_layer_TA(layer_TA layer, char *window);
