#ifndef PAR_TA_H
#define PAR_TA_H
#include "darknet_TA.h"
void aes_cbc_TA(char* xcrypt, float* gradient, int org_len);
void load_weights_TA(float *vec, int length, int layer_i, char type, int transpose);
void transpose_matrix_TA(float *a, int rows, int cols);

void save_weights_TA(float *weights_encrypted, int length, int layer_i, char type);
#endif
