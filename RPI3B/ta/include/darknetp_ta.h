#ifndef TA_DARKNETP_H
#define TA_DARKNETP_H

#include "darknet_TA.h"
#include "network_TA.h"

extern float *netta_truth;
extern int debug_summary_com;
extern int debug_summary_pass;
/*
 * This UUID is generated with uuidgen
 * the ITU-T UUID generator at http://www.itu.int/ITU-T/asn1/uuid.html
 */
#define TA_DARKNETP_UUID \
	{ 0x7fc5c039, 0x0542, 0x4ee1, \
		{ 0x80, 0xaf, 0xb4, 0xea, 0xb2, 0xf1, 0x99, 0x8d} }

/* The function IDs implemented in this TA */
#define MAKE_NETWORK_CMD 1
#define WORKSPACE_NETWORK_CMD 2
#define MAKE_CONV_CMD 3
#define MAKE_MAX_CMD 4
#define MAKE_DROP_CMD 5
#define MAKE_CONNECTED_CMD 6
#define MAKE_SOFTMAX_CMD 7
#define MAKE_COST_CMD 8
#define FORWARD_CMD 9
#define BACKWARD_CMD 10
#define BACKWARD_ADD_CMD 11
#define UPDATE_CMD 12
#define NET_TRUTH_CMD 13
#define CALC_LOSS_CMD 14
#define TRANS_WEI_CMD 15
#define OUTPUT_RETURN_CMD 16
#define SAVE_WEI_CMD 17

#define FORWARD_BACK_CMD 18
#define BACKWARD_BACK_CMD 19
#define BACKWARD_BACK_ADD_CMD 20

#define MAKE_AVG_CMD 21
#define FORWARD_CMD_TEST 22
#define FUSION_TEST 23

#define MAKE_ACTIVATE_CMD 24
#define FORWARD_CMD_NO_CONV_TEST 25
#define FORWARD_BACK_CMD_ONE_LAYER 26
#define FORWARD_CMD_RELU 27
#define INIT_CMD_MASK 28
#define FORWARD_CMD_PER_LAYER 29
#define FREE_CMD 30
#define FORWARD_CMD_PART 31
#define MAKE_SHORTCUT_LAYER_CMD 32
#define FIRST_N_CMD 33
#define FREE_N_CMD 34
#define RETURN_RUN_TIME_CMD 35
#define FORWARD_CMD_CONNECT_LAYER_PART 36
#define MAKE_BATCHNORM_CMD 37
void summary_array(const char *print_name, float *arr, int n);

#endif /*TA_DARKNETP_H*/
