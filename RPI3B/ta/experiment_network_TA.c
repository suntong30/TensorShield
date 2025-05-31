#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// #include <math.h>

#include "darknet_TA.h"
#include "blas_TA.h"
#include "network_TA.h"
#include "math_TA.h"
#include "experiment_network_TA.h"

#include "darknetp_ta.h"
#include <tee_internal_api.h>
#include <tee_internal_api_extensions.h>
#include <tee_api.h>
#include "activations_TA.h"
#include "convolutional_layer_TA.h"
#include "avgpool_layer_TA.h"
#include "maxpool_layer_TA.h"
#include "softmax_layer_TA.h"
#include "connected_layer_TA.h"
#include "dropout_layer_TA.h"
#include "cost_layer_TA.h"
#include "activation_layer_TA.h"
#include "parser_TA.h"

// network_TA netta;
// int roundnum = 0;
// float err_sum = 0;
// float avg_loss = -1;
// float *ta_net_input;
// float *ta_net_delta;
// float *ta_net_output;

static inline uint32_t tee_time_to_ms(TEE_Time t)
{
	return t.seconds * 1000 + t.millis;
}

static inline uint32_t get_delta_time_in_ms(TEE_Time start, TEE_Time stop)
{
	return tee_time_to_ms(stop) - tee_time_to_ms(start);
}
int netnum_part = 0;

void first_network(void)
{
    printf("[TEE] netlayers: %d\n", netta.n);
    printf("[TEE] first_network workspace: %d\n", netta.workspace_size);
    // if(netta.workspace_size){
    //     // printf("[TEE] first_network workspace: %d\n", netta.workspace_size);
    //     netta.workspace = calloc(1, netta.workspace_size);
    // }
    
}
void free_network(void)
{
    // if(netta.workspace_size){
    //     printf("[TEE] free_network workspace: %d\n", netta.workspace_size);
    //     free(netta.workspace);
    // }
}

// #define RUN_FUSION
// #define RUN_MASK
#define RUN_CAL
int forward_network_part_TA(int start_idx, int end_idx)
{
    int max_size_output = netta.layers[0].inputs;
#ifdef RUN_FUSION
    int * shuffle_idx = malloc(sizeof(int) * max_size_output);
    // other_read_raw_object(shuffle_idx, max_size_output);
    free(shuffle_idx);
    float *swap_temp_channel = malloc(sizeof(float) * max_size_output);
    int k = 1;
    for (int i = 0; i < max_size_output; ++i) {
        // printf("[eee]%d, %d\n", i, max_size_output);
        swap_temp_channel[((k + i) % max_size_output)] = netta.input[i];
    }
#ifndef RUN_MASK
    free(swap_temp_channel);
#endif
    // printf("[HELLO]\n");
#endif
#ifdef RUN_MASK
#ifndef RUN_FUSION
    float *swap_temp_channel = malloc(sizeof(float) * max_size_output);
#endif
    float *mask = malloc(sizeof(float) * max_size_output);

    other_read_raw_object(mask, max_size_output);
    for (int i = 0; i < max_size_output; ++i) {
        swap_temp_channel[i] -= mask[i];
    }
    free(mask);
    free(swap_temp_channel);
#endif
#ifdef RUN_CAL
    netnum_part = start_idx;
    // printf("[TEE] idx: %d, %d\n", start_idx, end_idx);
    while (netnum_part <= end_idx){
        layer_TA l = netta.layers[netnum_part];
        if (l.workspace_size > 0)
        {
            netta.workspace = (float*)calloc(1, l.workspace_size);
        }
        l.forward_TA(l, netta);
        netta.input = l.output;
        netnum_part++;
        if (l.workspace_size > 0)
        {
            free(netta.workspace);
        }
        
    }
    ta_net_output = netta.input;
    ta_net_input = netta.input;

#endif
    return 0;
}