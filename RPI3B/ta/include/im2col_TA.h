#ifndef IM2COL_TA_H
#define IM2COL_TA_H

void im2col_cpu_TA(float* data_im,
                int channels, int height, int width,
                int ksize, int stride, int pad, float* data_col);
void gemm_nn_TA_no_workspace(int M, int N, int K, float ALPHA,
             float *A, int lda,
             float *data_im, int ldb,
             float *C, int ldc,
             int ksize, int stride, int pad,
             int height, int width, int channels);
void gemm_nn_TA_no_workspace_sim(int M, int N, int K, float ALPHA,
             float *A, int lda,
             float *data_im, int ldb,
             float *C, int ldc,
             int ksize, int stride, int pad,
             int height, int width, int channels);
#endif
