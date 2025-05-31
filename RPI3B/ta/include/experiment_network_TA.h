#ifndef EXPERIMENT_NETWORK_TA_H
#define EXPERIMENT_NETWORK_TA_H
#include "darknet_TA.h"
// int forward_network_part_TA(int layer_forward_start_idx);
int forward_network_part_TA(int start_idx, int end_idx);
void first_network(void);
void free_network(void);
#endif
