#include "darknet_TA.h"

#include "batchnorm_layer_TA.h"
#include "blas_TA.h"
#include "math_TA.h"
#include "convolutional_layer_TA.h"
#include "stdio.h"

layer_TA make_batchnorm_layer_TA(int batch, int w, int h, int c)
{
    layer_TA l = {0};
    l.type = BATCHNORM_TA;
    l.batch = batch;
    l.h = l.out_h = h;
    l.w = l.out_w = w;
    l.c = l.out_c = c;
    l.output = (float*)calloc(h * w * c * batch, sizeof(float));
    if (l.output == NULL)
    {
        printf("[BATCHNORM] l.output failed to calloc\n");
    }
    // l.delta = calloc(h * w * c * batch, sizeof(float));
    l.outputs = l.inputs = w * h * c;
    
    l.scales = (float*)calloc(c, sizeof(float));
    if (l.scales == NULL)
    {
        printf("[BATCHNORM] l.scales failed to calloc\n");
    }
    // l.scale_updates = calloc(c, sizeof(float));
    l.biases = (float*)calloc(c, sizeof(float));
    if (l.biases == NULL)
    {
        printf("[BATCHNORM] l.biases failed to calloc\n");
    }
    // l.bias_updates = calloc(c, sizeof(float));
    // int i;
    // for(i = 0; i < c; ++i){
    //     l.scales[i] = 1;
    // }
    l.mean = (float*)calloc(c, sizeof(float));
    l.variance = (float*)calloc(c, sizeof(float));
    if (l.mean == NULL)
    {
        printf("[BATCHNORM] l.mean failed to calloc\n");
    }
    if (l.variance == NULL)
    {
        printf("[BATCHNORM] l.variance failed to calloc\n");
    }
    l.forward_TA = forward_batchnorm_layer_TA;
    // l.backward_TA = backward_batchnorm_layer_TA;
    l.rolling_mean = (float*)calloc(c, sizeof(float));
    l.rolling_variance = (float*)calloc(c, sizeof(float));

    l.x = (float *)calloc(l.batch * l.outputs, sizeof(float));
    // l.x_norm = (float *)calloc(l.batch * l.outputs, sizeof(float));
    if (l.rolling_mean == NULL)
    {
        printf("[BATCHNORM] l.rolling_mean failed to calloc\n");
    }

    if (l.rolling_variance == NULL)
    {
        printf("[BATCHNORM] l.rolling_variance failed to calloc\n");
    }
    
    return l;
}

void forward_batchnorm_layer_TA(layer_TA l, network_TA net)
{
    if(l.type == BATCHNORM_TA) copy_cpu_TA(l.outputs*l.batch, net.input, 1, l.output, 1);
    copy_cpu_TA(l.outputs*l.batch, l.output, 1, l.x, 1);
    if(net.train){
        mean_cpu_TA(l.output, l.batch, l.out_c, l.out_h*l.out_w, l.mean);
        variance_cpu_TA(l.output, l.mean, l.batch, l.out_c, l.out_h*l.out_w, l.variance);

        scal_cpu_TA(l.out_c, .99, l.rolling_mean, 1);
        axpy_cpu_TA(l.out_c, .01, l.mean, 1, l.rolling_mean, 1);
        scal_cpu_TA(l.out_c, .99, l.rolling_variance, 1);
        axpy_cpu_TA(l.out_c, .01, l.variance, 1, l.rolling_variance, 1);

        normalize_cpu_TA(l.output, l.mean, l.variance, l.batch, l.out_c, l.out_h*l.out_w);
        // copy_cpu_TA(l.outputs*l.batch, l.output, 1, l.x_norm, 1);
    } else {
        normalize_cpu_TA(l.output, l.rolling_mean, l.rolling_variance, l.batch, l.out_c, l.out_h*l.out_w);
    }
    scale_bias_TA(l.output, l.scales, l.batch, l.out_c, l.out_h*l.out_w);
    add_bias_TA(l.output, l.biases, l.batch, l.out_c, l.out_h*l.out_w);
    // printf("[TEE] batchnorm layer \n");
}

void backward_scale_cpu_TA(float *x_norm, float *delta, int batch, int n, int size, float *scale_updates)
{
    int i,b,f;
    for(f = 0; f < n; ++f){
        float sum = 0;
        for(b = 0; b < batch; ++b){
            for(i = 0; i < size; ++i){
                int index = i + size*(f + n*b);
                sum += delta[index] * x_norm[index];
            }
        }
        scale_updates[f] += sum;
    }
}

void mean_delta_cpu_TA(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta)
{

    int i,j,k;
    for(i = 0; i < filters; ++i){
        mean_delta[i] = 0;
        for (j = 0; j < batch; ++j) {
            for (k = 0; k < spatial; ++k) {
                int index = j*filters*spatial + i*spatial + k;
                mean_delta[i] += delta[index];
            }
        }
        mean_delta[i] *= (-1./ta_sqrt(variance[i] + .00001f));
    }
}


void  variance_delta_cpu_TA(float *x, float *delta, float *mean, float *variance, int batch, int filters, int spatial, float *variance_delta)
{

    int i,j,k;
    for(i = 0; i < filters; ++i){
        variance_delta[i] = 0;
        for(j = 0; j < batch; ++j){
            for(k = 0; k < spatial; ++k){
                int index = j*filters*spatial + i*spatial + k;
                variance_delta[i] += delta[index]*(x[index] - mean[i]);
            }
        }

        variance_delta[i] *= -.5 * ta_pow(variance[i] + .00001f, (float)(-3./2.));
    }
}

void normalize_delta_cpu_TA(float *x, float *mean, float *variance, float *mean_delta, float *variance_delta, int batch, int filters, int spatial, float *delta)
{
    int f, j, k;
    for(j = 0; j < batch; ++j){
        for(f = 0; f < filters; ++f){
            for(k = 0; k < spatial; ++k){
                int index = j*filters*spatial + f*spatial + k;
                delta[index] = delta[index] * 1./(ta_sqrt(variance[f] + .00001f)) + variance_delta[f] * 2. * (x[index] - mean[f]) / (spatial * batch) + mean_delta[f]/(spatial*batch);
            }
        }
    }
}


void backward_batchnorm_layer_TA(layer_TA l, network_TA net)
{
    if(!net.train){
        l.mean = l.rolling_mean;
        l.variance = l.rolling_variance;
    }
    backward_bias_TA(l.bias_updates, l.delta, l.batch, l.out_c, l.out_w*l.out_h);
    backward_scale_cpu_TA(l.x_norm, l.delta, l.batch, l.out_c, l.out_w*l.out_h, l.scale_updates);

    scale_bias_TA(l.delta, l.scales, l.batch, l.out_c, l.out_h*l.out_w);

    mean_delta_cpu_TA(l.delta, l.variance, l.batch, l.out_c, l.out_w*l.out_h, l.mean_delta);
    variance_delta_cpu_TA(l.x, l.delta, l.mean, l.variance, l.batch, l.out_c, l.out_w*l.out_h, l.variance_delta);
    normalize_delta_cpu_TA(l.x, l.mean, l.variance, l.mean_delta, l.variance_delta, l.batch, l.out_c, l.out_w*l.out_h, l.delta);
    if(l.type == BATCHNORM_TA) copy_cpu_TA(l.outputs*l.batch, l.delta, 1, net.delta, 1);
}
