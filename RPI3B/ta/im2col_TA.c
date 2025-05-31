#include "im2col_TA.h"
#include <stdio.h>

float im2col_get_pixel_TA(float *im, int height, int width, int channels,
                       int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;
    
    if (row < 0 || col < 0 ||
        row >= height || col >= width) return 0;
    return im[col + width*(row + height*channel)];
}

//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE
void im2col_cpu_TA(float* data_im,
                int channels,  int height,  int width,
                int ksize,  int stride, int pad, float* data_col)
{
    int c,h,w;
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;
    
    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                data_col[col_index] = im2col_get_pixel_TA(data_im, height, width, channels,
                                                       im_row, im_col, c_im, pad);
            }
        }
    }
}
void gemm_nn_TA_no_workspace(int M, int N, int K, float ALPHA,
             float *A, int lda,
             float *data_im, int ldb,
             float *C, int ldc,
             int ksize, int stride, int pad,
             int height, int width, int channels)
{
    int i,j,k;
    register int index = 0;
    register int height_col = (height + 2*pad - ksize) / stride + 1;
    register int width_col = (width + 2*pad - ksize) / stride + 1;
    register int im_row = 0;
    register int im_col;
    register int w_offset = 0;
    register int h_offset = 0;
    register int c_im = 0;
    register int c = 0;
    register int h = 0;
    register int w = 0;
    // for(i = 0; i < M; ++i){
    //     for(k = 0; k < K; ++k){
    //         register float A_PART = ALPHA*A[i*lda+k];
    //         for(j = 0; j < N; ++j){
    //             // index = k*ldb + j;
    //             // w = index % width_col;
    //             // h = index / width_col;
    //             // c = h / height_col;
    //             // h = h % height_col;
    //             // w_offset = c % ksize;
    //             // h_offset = (c / ksize) % ksize;
    //             // c_im = c / ksize / ksize;
    //             // im_row = h_offset + h * stride;
    //             // im_col = w_offset + w * stride;
    //             temp = im2col_get_pixel_TA(data_im, height, width, channels,
    //                                                    im_row, im_col, c_im, pad);
    //             C[i*ldc+j] += A_PART*temp;
    //             // C[i*ldc+j] += A_PART*B[k*ldb+j];
    //         }
    //     }
    // }
    register float A_PART;
    register float temp = 1.0;
    for(k = 0; k < K; ++k){
        for(j = 0; j < N; ++j){
            // index = k*ldb + j;
            // w = index % width_col;
            // h = index / width_col;
            // c = h / height_col;
            // h = h % height_col;
            // w_offset = c % ksize;
            // h_offset = (c / ksize) % ksize;
            // c_im = c / ksize / ksize;
            // im_row = h_offset + h * stride;
            // im_col = w_offset + w * stride;
            // temp = im2col_get_pixel_TA(data_im, height, width, channels,
            //                                         im_row, im_col, c_im, pad);
            for(i = 0; i < M; ++i){
                // register float A_PART = ALPHA*A[i*lda+k];
                A_PART = ALPHA*A[i*lda+k];
                C[i*ldc+j] += A_PART*temp;
                // C[i*ldc+j] += ALPHA*A[i*lda+k]*temp;
                // C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
    
}


void gemm_nn_TA_no_workspace_sim(int M, int N, int K, float ALPHA,
             float *A, int lda,
             float *data_im, int ldb,
             float *C, int ldc,
             int ksize, int stride, int pad,
             int height, int width, int channels)
{
    int i,j,k;
    int c,h,w;
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;
    
    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                float temp = im2col_get_pixel_TA(data_im, height, width, channels,
                                                       im_row, im_col, c_im, pad);
                
            }
        }
    }
    int div_size = N / (ksize * ksize);
    for(i = 0; i < M; ++i){
        // register int idx = (M - i - 1) * ldc;
        for(k = 0; k < K; ++k){
            register float A_PART = ALPHA*A[i*lda+k];
            for(j = 0; j < N; ++j){
                // C[i*ldc+j] += A_PART*B[k*ldb+j];
                // C[i*ldc+j] += A_PART*C[(M - i - 1)*ldc + j];
                C[i*ldc+j] += A_PART*data_im[k*div_size + j];
            }
        }
    }
    
}
void im2col_cpu_TA_low_workspace(float* data_im,
                int channels,  int height,  int width,
                int ksize,  int stride, int pad, float* data_col)
{
    int c,h,w;
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;
    
    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                data_col[col_index] = im2col_get_pixel_TA(data_im, height, width, channels,
                                                       im_row, im_col, c_im, pad);
            }
        }
    }
}
