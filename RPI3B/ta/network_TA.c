#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// #include <math.h>

#include "darknet_TA.h"
#include "blas_TA.h"
#include "network_TA.h"
#include "math_TA.h"

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
#include "shortcut_layer_TA.h"
network_TA netta;
int roundnum = 0;
float err_sum = 0;
float avg_loss = -1;

float *ta_net_input;
float *ta_net_delta;
float *ta_net_output;

void make_network_TA(int n, float learning_rate, float momentum, float decay, int time_steps, int notruth, int batch, int subdivisions, int random, int adam, float B1, float B2, float eps, int h, int w, int c, int inputs, int max_crop, int min_crop, float max_ratio, float min_ratio, int center, float clip, float angle, float aspect, float saturation, float exposure, float hue, int burn_in, float power, int max_batches)
{
    netta.n = n;

    //netta.seen = calloc(1, sizeof(size_t));
    netta.seen = calloc(1, sizeof(uint64_t));
    netta.layers = calloc(netta.n, sizeof(layer_TA));
    netta.t    = calloc(1, sizeof(int));
    netta.cost = calloc(1, sizeof(float));

    netta.learning_rate = learning_rate;
    netta.momentum = momentum;
    netta.decay = decay;
    netta.time_steps = time_steps;
    netta.notruth = notruth;
    netta.batch = batch;
    netta.subdivisions = subdivisions;
    netta.random = random;
    netta.adam = adam;
    netta.B1 = B1;
    netta.B2 = B2;
    netta.eps = eps;
    netta.h = h;
    netta.w = w;
    netta.c = c;
    netta.inputs = inputs;
    netta.max_crop = max_crop;
    netta.min_crop = min_crop;
    netta.max_ratio = max_ratio;
    netta.min_ratio = min_ratio;
    netta.center = center;
    netta.clip = clip;
    netta.angle = angle;
    netta.aspect = aspect;
    netta.saturation = saturation;
    netta.exposure = exposure;
    netta.hue = hue;
    netta.burn_in = burn_in;
    netta.power = power;
    netta.max_batches = max_batches;
    netta.workspace_size = 0;

    //netta.truth = net->truth; ////// ing network.c train_network
}
static inline uint32_t tee_time_to_ms(TEE_Time t)
{
	return t.seconds * 1000 + t.millis;
}

static inline uint32_t get_delta_time_in_ms(TEE_Time start, TEE_Time stop)
{
	return tee_time_to_ms(stop) - tee_time_to_ms(start);
}

void forward_network_TA()
{
    if(roundnum == 0){
        // ta_net_input malloc so not destroy before addition backward
        ta_net_input = malloc(sizeof(float) * netta.layers[0].inputs * netta.layers[0].batch);
        ta_net_delta = malloc(sizeof(float) * netta.layers[0].inputs * netta.layers[0].batch);
        // ta_net_input = malloc(sizeof(float) * netta.layers[0].inputs * netta.layers[0].batch);
        // ta_net_delta = malloc(sizeof(float) * netta.layers[0].inputs * netta.layers[0].batch);

        if(netta.workspace_size){
            IMSG("workspace_size=%ld\n", netta.workspace_size);
            netta.workspace = calloc(1, netta.workspace_size);
        }
    }

    roundnum++;
    int i;
    TEE_Time start_time = { };
	TEE_Time stop_time = { };
    TEE_GetREETime(&start_time);

    for(i = 0; i < netta.n; ++i){
        netta.index = i;
        layer_TA l = netta.layers[i];

        if(l.delta){
            fill_cpu_TA(l.outputs * l.batch, 0, l.delta, 1);
        }

        l.forward_TA(l, netta);

        // if(debug_summary_pass == 1){
        //     summary_array("forward_network / l.output", l.output, l.outputs*netta.batch);
        // }

        netta.input = l.output;

        if(l.truth) {
            netta.truth = l.output;
        }
        //output of the network (for predict)
        // &&
        // if(!netta.train && l.type == SOFTMAX_TA){
        //     ta_net_output = malloc(sizeof(float)*l.outputs*1);
        //     for(int z=0; z<l.outputs*1; z++){
        //         ta_net_output[z] = l.output[z];
        //     }
        // }

        if(i == netta.n - 1)  // ready to back REE for the rest forward pass
        {
            ta_net_input = malloc(sizeof(float)*l.outputs*l.batch);
            for(int z=0; z<l.outputs*l.batch; z++){
                ta_net_input[z] = netta.input[z];
            }
        }
    }
    TEE_GetREETime(&stop_time);
    IMSG("InVoked: start: %u.%u(s), stop: %u.%u(s), delta: %u(ms)",
			start_time.seconds, start_time.millis,
			stop_time.seconds, stop_time.millis,
			get_delta_time_in_ms(start_time, stop_time));
    // IMSG("InVoked: %lf seconds\n", sec(clock()-time));

    calc_network_cost_TA();
}
#define cluster_size (4)
#define OBJ_ID_1 "MASKMASK"
// extern char OBJ_ID_1[] = "MASKMASK";
#define OBJ_ID_SZ_1 8
#define OBJ_ID_2 "MASKMASKM"
// extern char OBJ_ID_2[] = "MASKMASKM";
#define OBJ_ID_SZ_2 9
// #define TEST_OBJECT_SIZE	(112 * 112 * 64)
#define TEST_OBJECT_SIZE	(64*64*64)
// #define 
extern int mask_chosen = 0;


#define OBJ_MASK_TEMPLATE "maskabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz"
#define MAX_MASK_BYTES (56)
#define WEIGHTS_INV_TEMPLATE "invabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz"
#define MAX_INV_BYTES (55)
#define DESHUFFLE_TEMPLATE "shuffleabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz"
#define MAX_DESHUFFLE_BYTES (59)

TEE_Result create_raw_object(char* obj_id, size_t obj_id_sz, float *load_data, int size)
{
	TEE_ObjectHandle object;
	TEE_Result res;
	uint32_t obj_data_flag;
	/*
	 * Create object in secure storage and fill with data
	 */
	obj_data_flag = TEE_DATA_FLAG_ACCESS_READ |		/* we can later read the oject */
			TEE_DATA_FLAG_ACCESS_WRITE |		/* we can later write into the object */
			TEE_DATA_FLAG_ACCESS_WRITE_META |	/* we can later destroy or rename the object */
			TEE_DATA_FLAG_OVERWRITE;		/* destroy existing object of same ID */


    // printf("tee_create_new_object\n");
	res = TEE_CreatePersistentObject(TEE_STORAGE_PRIVATE,
					obj_id, obj_id_sz,
					obj_data_flag,
					TEE_HANDLE_NULL,
					NULL, 0,		/* we may not fill it right now */
					&object);
    // printf("tee_create_new_object1\n");
	if (res != TEE_SUCCESS) {
		printf("TEE_CreatePersistentObject failed 0x%08x\n", res);
		// TEE_Free(obj_id);
		return res;
	}
    // printf("tee_create_new_object2\n");

	res = TEE_WriteObjectData(object, load_data, sizeof(float) * size);
	if (res != TEE_SUCCESS) {
		printf("TEE_WriteObjectData failed 0x%08x\n", res);
		TEE_CloseAndDeletePersistentObject1(object);
	} else {
		TEE_CloseObject(object);
	}
    // printf("tee_create_new_object3\n");

	// TEE_Free(obj_id);
	return res;
}

TEE_Result read_raw_object(float *load_data, int size){
    TEE_ObjectHandle object;
	TEE_ObjectInfo object_info;
	TEE_Result res;
	uint32_t read_bytes;
    if (mask_chosen == 0){
        res = TEE_OpenPersistentObject(TEE_STORAGE_PRIVATE,
                    OBJ_ID_1, OBJ_ID_SZ_1,
                    TEE_DATA_FLAG_ACCESS_READ |
                    TEE_DATA_FLAG_SHARE_READ,
                    &object);        
    }
    else{
        res = TEE_OpenPersistentObject(TEE_STORAGE_PRIVATE,
            OBJ_ID_2, OBJ_ID_SZ_2,
            TEE_DATA_FLAG_ACCESS_READ |
            TEE_DATA_FLAG_SHARE_READ,
            &object);    
    }

    if (res != TEE_SUCCESS) {
		printf("Failed to open persistent object, res=0x%08x\n", res);
		return res;
	}
	//     TEE_GetREETime(&start_time);

    res = TEE_GetObjectInfo1(object, &object_info);
	if (res != TEE_SUCCESS) {
		printf("Failed to create persistent object, res=0x%08x\n", res);
		return res;
	}

    res = TEE_ReadObjectData(object, load_data, sizeof(float) * size,
                &read_bytes);
	if (res != TEE_SUCCESS) {
		printf("TEE_ReadObjectData failed 0x%08x, read %" PRIu32 " over %u\n",
				res, read_bytes, object_info.dataSize);
        return res;
    }
    TEE_CloseObject(object);
    return res;
}
const int TEST_OBJECT_SIZE_CONST = TEST_OBJECT_SIZE;
void read_raw_object_total(float *load_data, int size)
{
    int num = size / TEST_OBJECT_SIZE_CONST;
    for (int i = 0; i < num; ++i)
    {
        read_raw_object(load_data + i * TEST_OBJECT_SIZE_CONST, TEST_OBJECT_SIZE_CONST);
    }
    int lest = size - num * TEST_OBJECT_SIZE_CONST;
    if (lest > 0)
    {
        read_raw_object(load_data + num * TEST_OBJECT_SIZE_CONST, lest);
    }
}
void other_read_raw_object(float *load_data, int size){
    read_raw_object(load_data, size);
}
float *simulate_array;

uint32_t mask_time_ms = 0;
uint32_t deobf_time_ms = 0;
uint32_t tee_operator_ms = 0;
uint32_t mask_time_ms_flops = 0;
uint32_t deobf_time_ms_flops = 0;
uint32_t relu_time_ms_flops = 0;
int max_size = 0;

int weights_length_array[300];
int current_idx_weights = 0;
void dechiper(char *dechiper_buffer, int size);
void free_simulate_arry(){
    // free(simulate_array);
    TEE_Time start_time = { };
	TEE_Time stop_time = { };
    // const int max_size_t = 224 * 224 * 64;
    int max_weights_length = -1;
    for (int _i = 0; _i < current_idx_weights; ++_i)
    {
        if(weights_length_array[_i] > max_weights_length)
        {
            max_weights_length = weights_length_array[_i];
        }
    }
    if (max_weights_length <= 0)
        return;
    float *simulate_relu = calloc(max_weights_length, sizeof(float));
    TEE_GetSystemTime(&start_time);
    for (int _i = 0; _i < current_idx_weights; ++_i)
    {
        dechiper(simulate_relu, weights_length_array[_i]);
    }
    TEE_GetSystemTime(&stop_time);
    printf("[TEE] simluate: dechiper: %u(ms)\n", get_delta_time_in_ms(start_time, stop_time));

    // float *simulate_relu = calloc(max_size_t, sizeof(float));
    // if (simulate_relu == NULL)
    // {
    //     printf("[TEE] simulate relu malloc error\n");
    //     return;
    // }
    // TEE_GetSystemTime(&start_time);
    // read_raw_object_total(simulate_relu, max_size_t);
    // for (int i = 0; i < max_size_t; ++i)
    // {
    //     simulate_relu[i] = simulate_relu[i] - simulate_relu[i / 2];
    // }
    
    // read_raw_object_total(simulate_relu, max_size_t);
    // for (int i = 0; i < max_size_t; ++i)
    // {
    //     simulate_relu[i] = simulate_relu[i] + simulate_relu[i / 2];
    // }
    // TEE_GetSystemTime(&stop_time);
    // printf("[TEE] simluate: cost_time: %u(ms)\n", get_delta_time_in_ms(start_time, stop_time));

    free(simulate_relu);
}
void write_mask_test(float *obj1_data, int size){

    int x = 1;
    // float *obj1_data = malloc(sizeof(float) * size);
    for(int i = 0; i < size; ++i) {
        obj1_data[i] = (1 / (i + 7));
    }
    create_raw_object(OBJ_ID_1, OBJ_ID_SZ_1, obj1_data, size);

    for(int i = 0; i < size; ++i) {
        obj1_data[i] = (1 / (i + 10));
    }
    create_raw_object(OBJ_ID_2, OBJ_ID_SZ_2, obj1_data, size);
    // free(obj1_data);
    // simulate_array = malloc(sizeof(float) * size * 8);
    // if (simulate_array == NULL){
    //     printf("[TEE] simulate_array malloc error\n");
    // }
    mask_time_ms = 0;
    deobf_time_ms = 0;
    tee_operator_ms = 0;
    free_simulate_arry();
}
void write_mask_test_test(float *obj1_data, int size){
    printf("[init_write_test]0\n");
    int x = 1;
    printf("[init_write_test]00\n");
    // float *obj1_data = malloc(sizeof(float) * size);
    printf("[init_write_test]1\n");
    printf("%d %d\n", size, TEST_OBJECT_SIZE);
    for(int i = 0; i < size; ++i) {
        obj1_data[i] = (1 / (i + 8));
    }
    printf("[init_write_test]2\n");

    create_raw_object(OBJ_ID_1, OBJ_ID_SZ_1, obj1_data, size);
    printf("[init_write_test]3\n");
    for(int i = 0; i < size; ++i) {
        obj1_data[i] = (1 / (i + 10));
    }
    printf("[init_write_test]4\n");
    create_raw_object(OBJ_ID_2, OBJ_ID_SZ_2, obj1_data, size);
    printf("CREATE MASK FINISH\n");
    // free(obj1_data);
}

const int mask_relu_max = 112 * 112 * 128 + 1;
// #define PRINT_TIME
#define RUN_FUSION
#define RUN_MASK
#define RUN_CAL
void tt_forward_relu(float *output, int size, int activation, int out_channels, int size_y, int use_mask, int flops){

    int max_size_output_ = size;
    TEE_Time start_time = { };
	TEE_Time stop_time = { };
    TEE_Time start_time_mask_2 = { };
	TEE_Time stop_time_mask_2 = { };
    TEE_Time start_time_mask_1 = { };
	TEE_Time stop_time_mask_1 = { };
    mask_time_ms_flops = 0;
    deobf_time_ms_flops = 0;
    relu_time_ms_flops = 0;
#ifdef RUN_FUSION
#ifdef PRINT_TIME
    printf("==============================\n");
    printf("[TEE] size: %d, size_y:%d, out_channels:%d, flops:%d \n", size, size_y, out_channels, flops);
    TEE_GetSystemTime(&start_time);
#endif

    float *temp_output_per_channel_y = malloc(sizeof(float) * size_y * size_y);
    // float *random_coeff_list_inv = malloc(sizeof(float) * out_channels * cluster_size);
    float *random_coeff_list_inv = calloc(((int)(out_channels / cluster_size)) * cluster_size * cluster_size, sizeof(float));
    // read_raw_object(random_coeff_list_inv, out_channels * cluster_size);
	if (temp_output_per_channel_y == NULL)
    {
        printf("[TEE] temp_output_per_channel_y malloc error\n");
    }
    if (random_coeff_list_inv == NULL)
    {
        printf("[TEE] random_coeff_list_inv malloc error\n");
    }
    for (int i = 0; i < out_channels / cluster_size; ++i){
        for(int j = 0; j < cluster_size; ++j) {
            for(int z = 0; z < size_y * size_y; ++z) {
                temp_output_per_channel_y[z] = 0;
                for (int k = 0; k  < cluster_size; ++k) {
                    temp_output_per_channel_y[z] += random_coeff_list_inv[i * cluster_size * cluster_size + j * cluster_size + k] * output[(i * cluster_size + k) * size_y * size_y + z];
                }
            }
            for (int z = 0; z < size_y * size_y; ++z) {
                // simulate_array[(i * cluster_size + j) * size_y * size_y + z] = temp_output_per_channel_y[z];
                output[(i * cluster_size + j) * size_y * size_y + z] = temp_output_per_channel_y[z];
            }
        }
    }
#ifdef PRINT_TIME_DEBUG
    printf("[TEE] inv mat\n");
#endif
    free(temp_output_per_channel_y);
    free(random_coeff_list_inv);
    temp_output_per_channel_y = NULL;
    random_coeff_list_inv = NULL;
    int * shuffle_idx = malloc(sizeof(int) * out_channels);
    // read_raw_object(shuffle_idx, out_channels);
    float *swap_temp_channel = malloc(sizeof(float) * max_size_output_);
    if(shuffle_idx == NULL)
    {
        printf("[TEE] shuffleidx malloc error\n");
    }
    if (swap_temp_channel == NULL)
    {
        printf("[TEE] swap temp channel malloc error\n");
    }
    int k = 1;

    for (int i = 0; i < out_channels; ++i) {
        for (int j = 0; j < size_y * size_y; ++j){
            swap_temp_channel[((k + i) % out_channels) * size_y * size_y + j] = output[i * size_y * size_y + j];
        }
    }
#ifdef PRINT_TIME_DEBUG
    printf("[TEE] swap channel\n");
#endif
    for (int i = 0; i < out_channels; ++i) {
        for (int j = 0; j < size_y * size_y; ++j){
            output[i * size_y * size_y + j] = swap_temp_channel[i * size_y * size_y + j];
            // simulate_array[i * size_y * size_y + j] = swap_temp_channel[i * size_y * size_y + j];
        }
    }
    free(shuffle_idx);
    free(swap_temp_channel);
    shuffle_idx = NULL;
    swap_temp_channel = NULL;
#ifdef PRINT_TIME
    TEE_GetSystemTime(&stop_time);
    deobf_time_ms_flops = get_delta_time_in_ms(start_time, stop_time);
    printf("[TEE] obj: %u(ms)\n", get_delta_time_in_ms(start_time, stop_time));
#endif
#endif
#ifdef PRINT_TIME_DEBUG
    printf("[TEE] del mask\n");
#endif
#ifdef RUN_MASK
#ifdef PRINT_TIME
    TEE_GetSystemTime(&start_time_mask_1);

#endif
    float *load_data;
    if(max_size_output_ < mask_relu_max)
    {
        load_data = malloc(sizeof(float) * max_size_output_);
        if (load_data == NULL)
        {
            printf("[TEE] load_data malloc error\n");
        }
        read_raw_object_total(load_data, max_size_output_);
        if (use_mask > 0){
            for (int i = 0; i < max_size_output_; ++i) {
                output[i] -= load_data[i];
                // simulate_array[i] -= load_data[i];
            }
        }
    }
#endif
#ifdef PRINT_TIME_DEBUG
    printf("[TEE] relu\n");
#endif
#ifdef RUN_CAL
#ifdef PRINT_TIME
    TEE_GetSystemTime(&stop_time_mask_1);
#endif
    activate_array_TA(output, size, (ACTIVATION_TA) activation);

#ifdef PRINT_TIME
    TEE_GetSystemTime(&stop_time);
    relu_time_ms_flops = get_delta_time_in_ms(start_time, stop_time);
    printf("[TEE] relu: %u(ms)\n", get_delta_time_in_ms(start_time, stop_time));
#endif
#endif
#ifdef RUN_MASK
#ifdef PRINT_TIME
    TEE_GetSystemTime(&start_time_mask_2);
#endif
#ifdef PRINT_TIME_DEBUG
    printf("[TEE] add mask\n");
#endif
    mask_chosen = rand_int();
#ifdef PRINT_TIME_DEBUG
    printf("[TEE] add mask-1\n");
#endif
    if (max_size_output_ < mask_relu_max)
    {
        read_raw_object_total(load_data, max_size_output_);
    #ifdef PRINT_TIME_DEBUG
        printf("[TEE] add mask-2\n");
    #endif
        for (int i = 0; i < max_size_output_; ++i) {
            // simulate_array[i] += load_data[i];
            output[i] += load_data[i];
        }
        free(load_data);
        load_data = NULL;
    }
#ifdef PRINT_TIME
    TEE_GetSystemTime(&stop_time_mask_2);
    mask_time_ms_flops = get_delta_time_in_ms(start_time_mask_2, stop_time_mask_2) + get_delta_time_in_ms(start_time_mask_1, stop_time_mask_1);
    printf("[TEE] mask: %u(ms)\n", get_delta_time_in_ms(start_time_mask_2, stop_time_mask_2) + get_delta_time_in_ms(start_time_mask_1, stop_time_mask_1));
#endif
#endif
#ifdef PRINT_TIME_DEBUG
    printf("[TEE] finished\n");
#endif
}
void tt_forward_network_TA(int n_layer_th)
{
    TEE_Time start_time = { };
	TEE_Time stop_time = { };
    TEE_GetREETime(&start_time);

    netta.index = n_layer_th;
    layer_TA l = netta.layers[n_layer_th];

    if(l.delta){
        fill_cpu_TA(l.outputs * l.batch, 0, l.delta, 1);
    }

    l.forward_TA(l, netta);
    netta.input = l.output;
    // ta_net_input = malloc(sizeof(float)*l.outputs*l.batch);
    // for(int z=0; z<l.outputs*l.batch; z++){
    //     ta_net_input[z] = netta.input[z];
    // }
    TEE_GetREETime(&stop_time);

    IMSG("InVoked: start: %u.%u(s), stop: %u.%u(s), delta: %u(ms)",
			start_time.seconds, start_time.millis,
			stop_time.seconds, stop_time.millis,
			get_delta_time_in_ms(start_time, stop_time));
    
}

// 交换两行
void swapRows(float matrix[cluster_size][cluster_size * 2], int row1, int row2) {
    for (int i = 0; i < cluster_size * 2; i++) {
        float temp = matrix[row1][i];
        matrix[row1][i] = matrix[row2][i];
        matrix[row2][i] = temp;
    }
}

// 计算矩阵的逆
int invertMatrix(float *input, float inverse[cluster_size][cluster_size]) {
    float augmented[cluster_size][2 * cluster_size];
    
    // 初始化增广矩阵
    for (int i = 0; i < cluster_size; i++) {
        for (int j = 0; j < cluster_size; j++) {
            augmented[i][j] = *(input + i * cluster_size + j);
        }
        for (int j = cluster_size; j < 2 * cluster_size; j++) {
            augmented[i][j] = (i == (j - cluster_size)) ? 1.0 : 0.0;
        }
    }

    // 进行高斯-约旦消元法
    for (int i = 0; i < cluster_size; i++) {
        // 寻找主元
        if (augmented[i][i] == 0) {
            int swapRow = i;
            for (int j = i + 1; j < cluster_size; j++) {
                if (augmented[j][i] != 0) {
                    swapRow = j;
                    break;
                }
            }
            if (augmented[swapRow][i] == 0) {
                // 无法求逆
                return 0;
            }
            swapRows(augmented, i, swapRow);
        }

        // 归一化主元行
        float pivot = augmented[i][i];
        for (int j = 0; j < 2 * cluster_size; j++) {
            augmented[i][j] /= pivot;
        }

        // 消去其他行的元素
        for (int j = 0; j < cluster_size; j++) {
            if (j != i) {
                float factor = augmented[j][i];
                for (int k = 0; k < 2 * cluster_size; k++) {
                    augmented[j][k] -= factor * augmented[i][k];
                }
            }
        }
    }

    // 提取逆矩阵
    for (int i = 0; i < cluster_size; i++) {
        for (int j = 0; j < cluster_size; j++) {
            inverse[i][j] = augmented[i][j + cluster_size];
        }
    }

    return 1;
}

// obfuscation
void forward_network_TA_DEDefusion(int out_channels, int size_y)
{
    // float
    // out_channels = 16;
    if (out_channels % 4 > 0)
        return;
    float *random_coeff_list = malloc(sizeof(float) * out_channels * cluster_size);
    float *random_coeff_list_inv = malloc(sizeof(float) * out_channels * cluster_size);
    for(int i = 0; i < out_channels * cluster_size; ++i) {
        random_coeff_list[i] = (float)(1 / (i + 7));
        // IMSG("%lf ", random_coeff_list[i]);
    }
    float *pre_matrix;
    float inv_matrix[cluster_size][cluster_size];
    int ret = 0;
    
    // for (int i = 0; i < (int)(out_channels / cluster_size); ++i) {
    //     pre_matrix = &random_coeff_list[i * cluster_size * cluster_size];
    //     ret = invertMatrix(pre_matrix, inv_matrix);
    //     for(int j = 0; j < cluster_size * cluster_size; ++j){
    //         // IMSG("r: %d\n", j);
    //         random_coeff_list_inv[i * cluster_size * cluster_size + j] = *(&inv_matrix[0][0] + j);
    //     }
    // }

    float *output_y = malloc(sizeof(float) * out_channels * size_y * size_y);
    for (int i = 0; i < out_channels * size_y * size_y; ++i)
        output_y[i] = (float)(1 / (i + 7));
    IMSG("output_y[0]: %lf\n", output_y[0]);
    float *temp_output_per_channel_y = malloc(sizeof(float) * size_y * size_y);
    // defusion
    TEE_Time start_time_mask = { };
    TEE_Time middle_time_mask = { };
	TEE_Time stop_time_mask = { };
    float *output_y_mask = malloc(sizeof(float) * out_channels * size_y * size_y);
    TEE_GetREETime(&start_time_mask);
    // generate mask
    int MAX_TIMEs = 20;
    for (int j = 0; j < MAX_TIMEs; ++j){
        for (int i = 0; i < out_channels * size_y * size_y; ++i)
            output_y_mask[i] = (float)(1 / (i + 7));
    }
    // add mask 
    TEE_GetREETime(&stop_time_mask);
    
    IMSG("[MASK (RAND) COST]: start: %u.%u(s), stop: %u.%u(s), delta: %u(us)\n",
			start_time_mask.seconds, start_time_mask.millis,
			stop_time_mask.seconds, stop_time_mask.millis,
			get_delta_time_in_ms(start_time_mask, stop_time_mask) * 1000 / MAX_TIMEs);
    TEE_GetREETime(&start_time_mask);

    for (int j = 0; j < MAX_TIMEs; ++j) {
        for (int i = 0; i < out_channels * size_y * size_y; ++i)
            output_y[i] = output_y[i] + output_y_mask[i];
        // remove mask
        for (int i = 0; i < out_channels * size_y * size_y; ++i)
            output_y[i] = output_y[i] - output_y_mask[i];
    }
    TEE_GetREETime(&stop_time_mask);
    IMSG("[MASK (ADD + DEL) COST]: start: %u.%u(s), stop: %u.%u(s), delta: %u(us)\n",
			start_time_mask.seconds, start_time_mask.millis,
			stop_time_mask.seconds, stop_time_mask.millis,
			get_delta_time_in_ms(start_time_mask, stop_time_mask) * 1000 / MAX_TIMEs);
    TEE_Time start_time = { };
	TEE_Time stop_time = { };

    

    IMSG("start: out_channels: %d, cluster_size: %d, size_y: %d\n", out_channels, cluster_size, size_y);
    TEE_GetREETime(&start_time);
    // start_time = clock();
    MAX_TIMEs = 1000;
    for (int ii = 0; ii < MAX_TIMEs; ++ii) {
        for (int i = 0; i < out_channels / cluster_size; ++i){
            for(int j = 0; j < cluster_size; ++j) {
                for(int z = 0; z < size_y * size_y; ++z) {
                    temp_output_per_channel_y[z] = 0;
                    for (int k = 0; k  < cluster_size; ++k) {
                        // temp_output_per_channel_y[z] += random_coeff_list_inv[i * cluster_size * cluster_size + j * cluster_size + k] * output_y[(i * cluster_size + k) * size_y * size_y + z];
                        temp_output_per_channel_y[z] += random_coeff_list[i * cluster_size * cluster_size + j * cluster_size + k] * output_y[(i * cluster_size + k) * size_y * size_y + z];
                    }
                }
                for (int z = 0; z < size_y * size_y; ++z) {
                    output_y[(i * cluster_size + j) * size_y * size_y + z] = temp_output_per_channel_y[z];
                }
            }
        }
    }    
    // stop_time = clock();
    TEE_GetREETime(&stop_time);
    // IMSG("defusion: %lf(s)", sec(stop_time - start_time));
    IMSG("[Defusion]: start: %u.%u(s), stop: %u.%u(s), delta: %u(us)\n",
			start_time.seconds, start_time.millis,
			stop_time.seconds, stop_time.millis,
			get_delta_time_in_ms(start_time, stop_time) * 1000 / MAX_TIMEs);
    free(random_coeff_list);
    free(random_coeff_list_inv);
    free(output_y);
    free(temp_output_per_channel_y);
    free(output_y_mask);
    // free()
}

// layer_TA only_one_layer;
float *only_one_input;
network_TA only_one_net;

#define TA_AES_ALGO_ECB			0
#define TA_AES_ALGO_CBC			1
#define TA_AES_ALGO_CTR			2

#define TA_AES_SIZE_128BIT		(128 / 8)
#define AES128_KEY_BIT_SIZE		128
#define AES128_KEY_BYTE_SIZE		(AES128_KEY_BIT_SIZE / 8)
struct aes_cipher {
	uint32_t algo;			/* AES flavour */
	uint32_t mode;			/* Encode or decode */
	uint32_t key_size;		/* AES key size in byte */
	TEE_OperationHandle op_handle;	/* AES ciphering operation */
	TEE_ObjectHandle key_handle;	/* transient object to load the key */
};
static TEE_Result ta2tee_algo_id(uint32_t param, uint32_t *algo)
{
	switch (param) {
	case TA_AES_ALGO_ECB:
		*algo = TEE_ALG_AES_ECB_NOPAD;
		return TEE_SUCCESS;
	case TA_AES_ALGO_CBC:
		*algo = TEE_ALG_AES_CBC_NOPAD;
		return TEE_SUCCESS;
	case TA_AES_ALGO_CTR:
		*algo = TEE_ALG_AES_CTR;
		return TEE_SUCCESS;
	default:
		EMSG("Invalid algo %u", param);
		return TEE_ERROR_BAD_PARAMETERS;
	}
}
static TEE_Result ta2tee_key_size(uint32_t param, uint32_t *key_size)
{
	switch (param) {
	case AES128_KEY_BYTE_SIZE:
		*key_size = param;
		return TEE_SUCCESS;
	default:
		EMSG("Invalid key size %u", param);
		return TEE_ERROR_BAD_PARAMETERS;
	}
}
static TEE_Result ta2tee_mode_id(uint32_t param, uint32_t *mode)
{
	switch (param) {
	case 1:
		*mode = TEE_MODE_ENCRYPT;
		return TEE_SUCCESS;
	case 0:
		*mode = TEE_MODE_DECRYPT;
		return TEE_SUCCESS;
	default:
		EMSG("Invalid mode %u", param);
		return TEE_ERROR_BAD_PARAMETERS;
	}
}

uint32_t *mode = TEE_MODE_DECRYPT;
static TEE_Result alloc_resources(struct aes_cipher *session)
{
	struct aes_cipher *sess;
	TEE_Attribute attr;
	TEE_Result res;
	char *key;

	/* Get ciphering context from session ID */
	sess = (struct aes_cipher *)session;

	res = ta2tee_algo_id(TA_AES_ALGO_CTR, &sess->algo);
	if (res != TEE_SUCCESS)
		return res;

	res = ta2tee_key_size(TA_AES_SIZE_128BIT, &sess->key_size);
	if (res != TEE_SUCCESS)
		return res;

	res = ta2tee_mode_id(0, &sess->mode);
	if (res != TEE_SUCCESS)
		return res;

	/*
	 * Ready to allocate the resources which are:
	 * - an operation handle, for an AES ciphering of given configuration
	 * - a transient object that will be use to load the key materials
	 *   into the AES ciphering operation.
	 */

	/* Free potential previous operation */
	if (sess->op_handle != TEE_HANDLE_NULL)
		TEE_FreeOperation(sess->op_handle);

	/* Allocate operation: AES/CTR, mode and size from params */
	res = TEE_AllocateOperation(&sess->op_handle,
				    sess->algo,
				    sess->mode,
				    sess->key_size * 8);
	if (res != TEE_SUCCESS) {
		EMSG("Failed to allocate operation");
		sess->op_handle = TEE_HANDLE_NULL;
		goto err;
	}

	/* Free potential previous transient object */
	if (sess->key_handle != TEE_HANDLE_NULL)
		TEE_FreeTransientObject(sess->key_handle);

	/* Allocate transient object according to target key size */
	res = TEE_AllocateTransientObject(TEE_TYPE_AES,
					  sess->key_size * 8,
					  &sess->key_handle);
	if (res != TEE_SUCCESS) {
		EMSG("Failed to allocate transient object");
		sess->key_handle = TEE_HANDLE_NULL;
		goto err;
	}

	/*
	 * When loading a key in the cipher session, set_aes_key()
	 * will reset the operation and load a key. But we cannot
	 * reset and operation that has no key yet (GPD TEE Internal
	 * Core API Specification – Public Release v1.1.1, section
	 * 6.2.5 TEE_ResetOperation). In consequence, we will load a
	 * dummy key in the operation so that operation can be reset
	 * when updating the key.
	 */
	key = TEE_Malloc(sess->key_size, 0);
	if (!key) {
		res = TEE_ERROR_OUT_OF_MEMORY;
		goto err;
	}

	TEE_InitRefAttribute(&attr, TEE_ATTR_SECRET_VALUE, key, sess->key_size);

	res = TEE_PopulateTransientObject(sess->key_handle, &attr, 1);
	if (res != TEE_SUCCESS) {
		EMSG("TEE_PopulateTransientObject failed, %x", res);
		goto err;
	}

	res = TEE_SetOperationKey(sess->op_handle, sess->key_handle);
	if (res != TEE_SUCCESS) {
		EMSG("TEE_SetOperationKey failed %x", res);
		goto err;
	}
	return res;

err:
	if (sess->op_handle != TEE_HANDLE_NULL)
		TEE_FreeOperation(sess->op_handle);
	sess->op_handle = TEE_HANDLE_NULL;
	if (sess->key_handle != TEE_HANDLE_NULL)
		TEE_FreeTransientObject(sess->key_handle);
	sess->key_handle = TEE_HANDLE_NULL;
	return res;
}

#define AES_TEST_KEY_SIZE			16
#define AES_BLOCK_SIZE				16
static TEE_Result set_aes_key(struct aes_cipher *session, char *key, uint32_t key_sz)
{
	struct aes_cipher *sess;
	TEE_Attribute attr;
	TEE_Result res;
	sess = (struct aes_cipher *)session;

	if (key_sz != sess->key_size) {
		EMSG("Wrong key size %" PRIu32 ", expect %" PRIu32 " bytes",
		     key_sz, sess->key_size);
		return TEE_ERROR_BAD_PARAMETERS;
	}


	TEE_InitRefAttribute(&attr, TEE_ATTR_SECRET_VALUE, key, key_sz);

	TEE_ResetTransientObject(sess->key_handle);
	res = TEE_PopulateTransientObject(sess->key_handle, &attr, 1);
	if (res != TEE_SUCCESS) {
		EMSG("TEE_PopulateTransientObject failed, %x", res);
		return res;
	}

	TEE_ResetOperation(sess->op_handle);
	res = TEE_SetOperationKey(sess->op_handle, sess->key_handle);
	if (res != TEE_SUCCESS) {
		EMSG("TEE_SetOperationKey failed %x", res);
		return res;
	}

	return res;
}
static TEE_Result reset_aes_iv(struct aes_cipher *session, char *iv)
{
	struct aes_cipher *sess;

	/* Get ciphering context from session ID */
	sess = (struct aes_cipher *)session;
	/*
	 * Init cipher operation with the initialization vector.
	 */
	TEE_CipherInit(sess->op_handle, iv, AES_BLOCK_SIZE);

	return TEE_SUCCESS;
}
static TEE_Result cipher_buffer(void *session, char* buffer, uint32_t sz)
{
	struct aes_cipher *sess;

	/* Get ciphering context from session ID */
	sess = (struct aes_cipher *)session;


	if (sess->op_handle == TEE_HANDLE_NULL)
		return TEE_ERROR_BAD_STATE;

	/*
	 * Process ciphering operation on provided buffers
	 */
    char *temp = malloc(sizeof(char) * sz);
    
    TEE_CipherUpdate(sess->op_handle,
				buffer, sz,
				temp, &sz);
    for(int i = 0; i < sz; ++i)
    {
        buffer[i] = temp[i];
    }
    free(temp);

    return TEE_SUCCESS;

}


void dechiper(char *dechiper_buffer, int size){
    struct aes_cipher *session;
    session = TEE_Malloc(sizeof(*session), 0);
    alloc_resources(session);
    char KEY[128] = {
        0xca, 0xaa, 0x20, 0x34, 0x4c, 0xa9, 0x91, 0x90, 0xc8, 0x53, 0x86, 0x28, 0xd4, 0x66,
        0xd3, 0xa6, 0x4e, 0xe5, 0x35, 0x4a, 0x34, 0xbd, 0x9f, 0x53, 0xe6, 0x58, 0xf7, 0x06, 0xf9,
        0x5d, 0x29, 0x80, 0xce, 0xc2, 0xa5, 0x9c, 0xa0, 0xe2, 0x63, 0x79, 0xc1, 0xdc, 0x17, 0x9d,
        0xfd, 0x95, 0x93, 0x3a, 0xbc, 0x9d, 0xd3, 0xf5, 0xfb, 0x7a, 0xee, 0x31, 0xb7, 0x39, 0x01,
        0x18, 0x87, 0xa7, 0x2c, 0x00, 0xd1, 0x2e, 0xfb, 0x04, 0xed, 0xa3, 0x92, 0xc9, 0xa2, 0x99,
        0xa6, 0x4d, 0x6b, 0x82, 0x14, 0x9a, 0x00, 0x06, 0xc6, 0x16, 0x56, 0x83, 0x48, 0xad, 0x27,
        0x0e, 0x27, 0x6b, 0x47, 0x2d, 0x84, 0xd4, 0xaf, 0xea, 0xb7, 0x03, 0x00, 0xbe, 0xcd, 0x91,
        0xd4, 0x33, 0x96, 0x41, 0x61, 0x43, 0xf9, 0x37, 0x18, 0x14, 0x67, 0xf0, 0x48, 0x68, 0xa1,
        0xfe, 0x50, 0x73, 0xda, 0xce, 0xbb, 0xff, 0x2e, 0x49
    };
    // char iv[AES_BLOCK_SIZE] = {0};
    set_aes_key(session, KEY, 16);

    char *iv = calloc(AES_BLOCK_SIZE, sizeof(char));
    reset_aes_iv(session, iv);
    cipher_buffer(session, dechiper_buffer, size);
    free(iv);

    iv = NULL;
    TEE_FreeTransientObject(session->key_handle);
	// if (session->key_handle != TEE_HANDLE_NULL)
    session->key_handle = TEE_HANDLE_NULL;
    TEE_FreeOperation(session->op_handle);
	// if (session->op_handle != TEE_HANDLE_NULL)
    session->op_handle = TEE_HANDLE_NULL;
    TEE_Free(session);
}
// #define USE_CHIPER
#define BLOCK_NUM (128)
void copy_output(layer_info_and_weights *layer_info, float *layer_output, int sz){
    only_one_net.input = NULL;
    if(layer_info->need_input <= 0) {
        free(only_one_input);
        only_one_input = NULL;
    }
    only_one_input = malloc(sizeof(float) * sz);
    if (only_one_input == NULL)
    {
        printf("[TEE] failed to malloc only one input\n");
    }
    copy_cpu_TA(sz, layer_output, 1, only_one_input, 1);
    // free(layer_output);
    // layer_output = NULL;
}
// int no_workspace = 1;
int no_workspace = 0;
const int max_workspace_size = (int) (1024 * 1024 * 4.5);
// const int max_workspace_size = 1024;
void run_conv_layer(layer_info_and_weights *layer_info, float *base_parameters, float *batch_parameters) {
#ifdef PRINT_TIME_DEBUG
    printf("[TEE] run_conv_layer: 0\n");
#endif
    if (layer_info->workspace_size >= max_workspace_size)
    {
        no_workspace = 1;
    }
    else 
    {
        no_workspace = 0;
    }
    // printf("[TEE]WORKSPACE:%d, max_workspace_size:%d, no_workspace:%d\n", layer_info->workspace_size, max_workspace_size, no_workspace);
    layer_TA only_one_layer = make_convolutional_layer_TA_new(layer_info->batch,
        layer_info->h, layer_info->w, layer_info->c, layer_info->n,
        layer_info->groups, layer_info->size, layer_info->stride,
        layer_info->padding, layer_info->activation, layer_info->batch_normalize,
        layer_info->binary, layer_info->xnor, layer_info->adam,
        layer_info->flipped, layer_info->dot);
    int flags_alloc = 0;
    if(only_one_layer.weights == NULL)
    {
        printf("[TEE] run_conv_layer weights calloc failed\n");
        // flags_alloc = 1;
        // only_one_layer.weights = base_parameters;
    }
#ifdef PRINT_TIME_DEBUG
    else
    {
        printf("[TEE] run_conv_layer weights calloc success, addr: 0x%x\n", only_one_layer.weights);

    }
#endif
    if(only_one_layer.biases == NULL)
    {
        printf("[TEE] run_conv_layer biases calloc failed\n");
    }

#ifdef PRINT_TIME_DEBUG
    else {
        printf("[TEE] run_conv_layer biases calloc success, addr: 0x%x\n", only_one_layer.biases);
    }
#endif

#ifdef PRINT_TIME_DEBUG
    printf("[TEE] run_conv_layer: 1\n");
#endif
    // decrypted operation
    // LOAD WEIGHTS
    if(layer_info->batch_normalize) {
        copy_cpu_TA(layer_info->n, batch_parameters + layer_info->n * 0, 1, only_one_layer.scales, 1);
        copy_cpu_TA(layer_info->n, batch_parameters + layer_info->n * 1, 1, only_one_layer.rolling_mean, 1);
        copy_cpu_TA(layer_info->n, batch_parameters + layer_info->n * 2, 1, only_one_layer.rolling_variance, 1);
    }
#ifdef PRINT_TIME_DEBUG
    printf("[TEE] run_conv_layer: 2, bias_length: %d, weight_length: %d, %d, %d\n", layer_info->biases_length, layer_info->weights_length, only_one_layer.nweights, only_one_layer.groups);
#endif

    copy_cpu_TA(only_one_layer.nweights, base_parameters, 1, only_one_layer.weights, 1);
    copy_cpu_TA(only_one_layer.nbiases, base_parameters + only_one_layer.nweights, 1, only_one_layer.biases, 1);
    // only_one_layer.weights = base_parameters;
    // only_one_layer.biases = base_parameters + only_one_layer.nweights;

#ifdef USE_CHIPER
    TEE_Time start_time = { };
    TEE_Time stop_time = { };
    TEE_GetREETime(&start_time);

    if (layer_info->weights_length <= 1024 * 1024 * 3 * 3){
        int base_block = 2048 * BLOCK_NUM;
        if (base_block < layer_info->weights_length){
            for (int i = 0; i < layer_info->weights_length / base_block; ++i){
                // printf("%d ", i);
                aes_cbc_TA("decrypt", &(only_one_layer.weights[i * base_block]), base_block);
                // dechiper(&(only_one_layer.weights[i * base_block]), base_block);
            }
            // printf("[tee] block: %d, %d, %d\n", base_block, layer_info->weights_length / base_block, layer_info->weights_length);
        }
        else {
            aes_cbc_TA("decrypt", &(only_one_layer.weights[0]), layer_info->weights_length);
        }
        // aes_cbc_TA("decrypt", only_one_layer.weights, layer_info->weights_length);
        aes_cbc_TA("decrypt", only_one_layer.biases, layer_info->biases_length);
        // dechiper(only_one_layer.weights, layer_info->weights_length);
        // dechiper(only_one_layer.biases, layer_info->biases_length);
    }
    TEE_GetREETime(&stop_time);
    // IMSG("defusion: %lf(s)", sec(stop_time - start_time));
    printf("[conv dechipher]: delta: %u(us)\n", get_delta_time_in_ms(start_time, stop_time));
    // IMSG("after_decrypt");
#endif
#ifdef PRINT_TIME_DEBUG
    printf("[TEE] run_conv_layer: 3\n");
#endif
    if(only_one_layer.flipped)
        transpose_matrix_TA(only_one_layer.weights, only_one_layer.c*only_one_layer.size*only_one_layer.size/only_one_layer.groups, only_one_layer.n);
#ifdef PRINT_TIME_DEBUG
    printf("[TEE] run_conv_layer: 4\n");
#endif
    only_one_net.workspace_size = only_one_layer.workspace_size;
    if (only_one_net.workspace_size > 0) {
        only_one_net.workspace = calloc(1, only_one_net.workspace_size);
        if(only_one_net.workspace == NULL) {
            printf("[TEE] workspace alloc error, %d.\n", only_one_net.workspace_size);
        }
    }
#ifdef PRINT_TIME_DEBUG
    printf("[TEE] run_conv_layer: 5, workspace_size: %d\n", only_one_net.workspace_size);
#endif
    only_one_layer.forward_TA(only_one_layer, only_one_net);
#ifdef PRINT_TIME_DEBUG
    printf("[TEE] run_conv_layer: 6\n");
#endif

    if (only_one_net.workspace_size > 0) {
        free(only_one_net.workspace);
        only_one_net.workspace = NULL;
    }
#ifdef PRINT_TIME_DEBUG
    printf("[TEE] run_conv_layer: 7\n");
#endif


    if(layer_info->batch_normalize) {
        free(only_one_layer.scales);
        free(only_one_layer.mean);
        free(only_one_layer.mean_delta);
        free(only_one_layer.variance);
        free(only_one_layer.variance_delta);
        free(only_one_layer.rolling_mean);
        free(only_one_layer.rolling_variance);
        free(only_one_layer.x);
        // free(only_one_layer.x_norm);
    }
    // IMSG("[TEE] run_conv_layer: 6\n");
#ifdef PRINT_TIME_DEBUG
    printf("[TEE] run_conv_layer: 8\n");
#endif

    if(only_one_layer.binary) {
        free(only_one_layer.binary_weights);
        free(only_one_layer.cweights);
        free(only_one_layer.scales);
    }
#ifdef PRINT_TIME_DEBUG
    printf("[TEE] run_conv_layer: 10\n");
#endif
    if(only_one_layer.xnor) {
        if(~only_one_layer.binary) {
            free(only_one_layer.binary_weights);
        }
        free(only_one_layer.binary_input);
    }
#ifdef PRINT_TIME_DEBUG
    printf("[TEE] run_conv_layer: 11\n");
#endif
    if (only_one_layer.m != NULL) {
        free(only_one_layer.m);
        free(only_one_layer.v);
        free(only_one_layer.bias_m);
        free(only_one_layer.scale_m);
        free(only_one_layer.bias_v);
        free(only_one_layer.scale_v);
    }
#ifdef PRINT_TIME_DEBUG
    printf("[TEE] run_conv_layer: 12\n");
#endif

    if(only_one_layer.biases != NULL)
        free(only_one_layer.biases);
    only_one_layer.biases = NULL;
#ifdef PRINT_TIME_DEBUG
    printf("[TEE] run_conv_layer: 9\n");
#endif
    if(only_one_layer.weights != NULL)
        free(only_one_layer.weights);
    only_one_layer.weights = NULL;
#ifdef PRINT_TIME_DEBUG
    printf("[TEE] run_conv_layer: 8-9\n");
#endif

    copy_output(layer_info, only_one_layer.output, only_one_layer.outputs);
#ifdef PRINT_TIME_DEBUG
    printf("[TEE] run_conv_layer: 13\n");
#endif
    free(only_one_layer.output);
    only_one_layer.output = NULL;
    // printf("[TEE] num: %d %d\n", only_one_layer.outputs * only_one_layer.batch, only_one_layer.batch);
}


int chunk_connected = 0;

void run_connected_layer(layer_info_and_weights *layer_info, float *base_parameters, float *batch_parameters) {
    if (layer_info->inputs * layer_info->outputs > 512 * 4096)
    {
        chunk_connected = 1;
        int num = layer_info->inputs / 512;
        for (int i = 1; i < num; ++i)
        {
            dechiper(base_parameters, layer_info->weights_length);
        }
        int lest = layer_info->inputs - num * 512;
        if (lest > 0)
        {
            num++;
            dechiper(base_parameters, lest);
        }
    }
    else
    {
        chunk_connected = 0;
    }
    layer_TA only_one_layer = make_connected_layer_TA_new(
        layer_info->batch, layer_info->inputs,
        layer_info->outputs, layer_info->activation,
        layer_info->batch_normalize, layer_info->adam
    );
    // decrypted operation
    // LOAD WEIGHTS
    if(layer_info->batch_normalize) {
        copy_cpu_TA(layer_info->outputs, batch_parameters + layer_info->outputs * 0, 1, only_one_layer.scales, 1);
        copy_cpu_TA(layer_info->outputs, batch_parameters + layer_info->outputs * 1, 1, only_one_layer.rolling_mean, 1);
        copy_cpu_TA(layer_info->outputs, batch_parameters + layer_info->outputs * 2, 1, only_one_layer.rolling_variance, 1);
    }
    copy_cpu_TA(layer_info->weights_length, base_parameters, 1, only_one_layer.weights, 1);
    copy_cpu_TA(layer_info->biases_length, base_parameters + layer_info->weights_length, 1, only_one_layer.biases, 1);

#ifdef USE_CHIPER
    if (layer_info->weights_length <= 1024 * 1024 * 3 * 3){
        int base_block = 2048 * BLOCK_NUM;
        if (base_block < layer_info->weights_length){
            for (int i = 0; i < layer_info->weights_length / base_block; ++i){
                aes_cbc_TA("decrypt", &(only_one_layer.weights[i * base_block]), base_block);
                // dechiper(&(only_one_layer.weights[i * base_block]), base_block);
            }
        }
        else {
            aes_cbc_TA("decrypt", &(only_one_layer.weights[0]), layer_info->weights_length);
        }
        // aes_cbc_TA("decrypt", only_one_layer.weights, layer_info->weights_length);
        aes_cbc_TA("decrypt", only_one_layer.biases, layer_info->biases_length);
        // dechiper(only_one_layer.weights, layer_info->weights_length);
        // dechiper(only_one_layer.biases, layer_info->biases_length);
    }
    // dechiper(only_one_layer.weights, layer_info->weights_length);
    // dechiper(only_one_layer.biases, layer_info->biases_length);
#endif
    if(only_one_layer.flipped)
        transpose_matrix_TA(only_one_layer.weights, only_one_layer.inputs, only_one_layer.outputs);
    only_one_net.workspace_size = only_one_layer.workspace_size;
    if (only_one_net.workspace_size > 0) {
        only_one_net.workspace = calloc(1, only_one_net.workspace_size);
        if(only_one_net.workspace == NULL) {
            printf("[TEE] workspace alloc error, %x.\n", only_one_net.workspace);
        }
    }
    only_one_layer.forward_TA(only_one_layer, only_one_net);
    if (only_one_net.workspace_size > 0) {
        free(only_one_net.workspace);
        only_one_net.workspace = NULL;
    }
    if(layer_info->batch_normalize) {
        free(only_one_layer.scales);
        free(only_one_layer.mean);
        free(only_one_layer.mean_delta);
        free(only_one_layer.variance);
        free(only_one_layer.variance_delta);
        free(only_one_layer.rolling_mean);
        free(only_one_layer.rolling_variance);
        free(only_one_layer.x);
        free(only_one_layer.x_norm);
        only_one_layer.scales = NULL;
        only_one_layer.mean = NULL;
        only_one_layer.mean_delta = NULL;
        only_one_layer.variance = NULL;
        only_one_layer.variance_delta = NULL;
        only_one_layer.rolling_mean = NULL;
        only_one_layer.rolling_variance = NULL;
        only_one_layer.x = NULL;
        only_one_layer.x_norm = NULL;
    }
    if (only_one_layer.m != NULL) {
        free(only_one_layer.m);
        free(only_one_layer.v);
        free(only_one_layer.bias_m);
        free(only_one_layer.scale_m);
        free(only_one_layer.bias_v);
        free(only_one_layer.scale_v);
        only_one_layer.m = NULL;
        only_one_layer.v = NULL;
        only_one_layer.bias_m = NULL;
        only_one_layer.scale_m = NULL;
        only_one_layer.bias_v = NULL;
        only_one_layer.scale_v = NULL;
    }
    free(only_one_layer.weights);
    only_one_layer.weights = NULL;
    free(only_one_layer.biases);
    only_one_layer.biases = NULL;
    copy_output(layer_info, only_one_layer.output, only_one_layer.outputs);
    free(only_one_layer.output);
    only_one_layer.output = NULL;

}
void run_maxpool_layer(layer_info_and_weights* layer_info) {
    layer_TA only_one_layer = make_maxpool_layer_TA(
        layer_info->batch, layer_info->h, layer_info->w, layer_info->c,
        layer_info->size, layer_info->stride, layer_info->padding
    );
    only_one_net.workspace_size = only_one_layer.workspace_size;
    if (only_one_net.workspace_size > 0) {
        only_one_net.workspace = calloc(1, only_one_net.workspace_size);
    }
    only_one_layer.forward_TA(only_one_layer, only_one_net);
    free(only_one_layer.indexes);
    only_one_layer.indexes = NULL;
    if (only_one_net.workspace_size > 0) {
        free(only_one_net.workspace);
        only_one_net.workspace = NULL;
    }
    copy_output(layer_info, only_one_layer.output, only_one_layer.outputs);
    free(only_one_layer.output);
    only_one_layer.output = NULL;
}
void run_avgpool_layer(layer_info_and_weights* layer_info) {
    layer_TA only_one_layer = make_avgpool_layer_TA(
        layer_info->batch, layer_info->h, layer_info->w, layer_info->c
    );
    only_one_net.workspace_size = only_one_layer.workspace_size;
    if (only_one_net.workspace_size > 0) {
        only_one_net.workspace = calloc(1, only_one_net.workspace_size);
    }
    only_one_layer.forward_TA(only_one_layer, only_one_net);
    if (only_one_net.workspace_size > 0) {
        free(only_one_net.workspace);
        only_one_net.workspace = NULL;
    }
    copy_output(layer_info, only_one_layer.output, only_one_layer.outputs);
    free(only_one_layer.output);
    only_one_layer.output = NULL;
}
void run_softmax_layer(layer_info_and_weights* layer_info) {
    layer_TA only_one_layer = make_softmax_layer_TA_new(
        layer_info->batch, layer_info->inputs, layer_info->groups, layer_info->temperature,
        layer_info->w, layer_info->h, layer_info->c, layer_info->spatial, layer_info->noloss
    );
    only_one_net.workspace_size = only_one_layer.workspace_size;
    if (only_one_net.workspace_size > 0) {
        only_one_net.workspace = calloc(1, only_one_net.workspace_size);
    }
    only_one_layer.forward_TA(only_one_layer, only_one_net);
    free(only_one_layer.loss);
    free(only_one_layer.cost);
    only_one_layer.loss = NULL;
    only_one_layer.cost = NULL;
    if (only_one_net.workspace_size > 0) {
        free(only_one_net.workspace);
        only_one_net.workspace = NULL;
    }
    copy_output(layer_info, only_one_layer.output, only_one_layer.outputs);
    free(only_one_layer.output);
    only_one_layer.output = NULL;
}
void run_dropout_layer(layer_info_and_weights* layer_info) {
    layer_TA only_one_layer = make_dropout_layer_TA_new(
        layer_info->batch, layer_info->inputs, layer_info->probability,
        layer_info->w, layer_info->h, layer_info->c, layer_info->netnum
    );
    only_one_net.workspace_size = only_one_layer.workspace_size;
    if (only_one_net.workspace_size > 0) {
        only_one_net.workspace = calloc(1, only_one_net.workspace_size);
    }
    only_one_layer.forward_TA(only_one_layer, only_one_net);
    if (only_one_net.workspace_size > 0) {
        free(only_one_net.workspace);
        only_one_net.workspace = NULL;
    }
    free(only_one_layer.rand);
    only_one_layer.rand = NULL;
    copy_output(layer_info, only_one_layer.output, only_one_layer.outputs);
    free(only_one_layer.output);
    only_one_layer.output = NULL;
}

void run_cost_layer(layer_info_and_weights* layer_info) {
    layer_TA only_one_layer = make_cost_layer_TA_new(
        layer_info->batch, layer_info->inputs,
        layer_info->cost_type, layer_info->scale,
        layer_info->ratio, layer_info->noobject_scale, layer_info->thresh
    );
    only_one_net.workspace_size = only_one_layer.workspace_size;
    if (only_one_net.workspace_size > 0) {
        only_one_net.workspace = calloc(1, only_one_net.workspace_size);
    }
    only_one_layer.forward_TA(only_one_layer, only_one_net);
    free(only_one_layer.cost);
    free(only_one_layer.delta);
    only_one_layer.cost = NULL;
    only_one_layer.delta = NULL;
    if (only_one_net.workspace_size > 0) {
        free(only_one_net.workspace);
        only_one_net.workspace = NULL;
    }
    copy_output(layer_info, only_one_layer.output, only_one_layer.outputs);
    free(only_one_layer.output);
    only_one_layer.output = NULL;
}
void run_shortcut_layer(layer_info_and_weights* layer_info) {

    layer_TA only_one_layer = make_shortcut_layer_TA(layer_info->batch,
        layer_info->index, layer_info->w, layer_info->h, layer_info->c,
        layer_info->w2, layer_info->h2, layer_info->c2);
    only_one_layer.forward_TA(only_one_layer, only_one_net);
    copy_output(layer_info, only_one_layer.output, only_one_layer.outputs);
    free(only_one_layer.output);
    free(only_one_layer.add);
    only_one_layer.output = NULL;
    only_one_layer.add = NULL;

    // for (int i = 0; i < layer_info->outputs; ++i) {
    //     only_one_input[i] = only_one_input[i] + only_one_input[i];
    // }
    // activate_array_TA(only_one_input, layer_info->outputs, RELU_TA);
    // copy_output(layer_info, only_one_layer.output, only_one_layer.outputs);
}   


layer_TA connect_one_layer;
void run_connected_layer_part(layer_info_and_weights *layer_info, float *base_parameters, float *batch_parameters) {
    if (chunk_connected == 0)
    {
        chunk_connected = 1;
        connect_one_layer = make_connected_layer_TA_new(
            layer_info->batch, layer_info->inputs,
            layer_info->outputs, layer_info->activation,
            layer_info->batch_normalize, layer_info->adam
        );

        if (layer_info->batch_normalize)
        {   
            if (layer_info->type == CONVOLUTIONAL_TA || layer_info->type == CONNECTED_TA)
            {
                dechiper(batch_parameters + layer_info->biases_length * 0, layer_info->biases_length);
                dechiper(batch_parameters + layer_info->biases_length * 1, layer_info->biases_length);
                dechiper(batch_parameters + layer_info->biases_length * 2, layer_info->biases_length);
            }
        }
        if(layer_info->batch_normalize) {
            copy_cpu_TA(layer_info->outputs, batch_parameters + layer_info->outputs * 0, 1, connect_one_layer.scales, 1);
            copy_cpu_TA(layer_info->outputs, batch_parameters + layer_info->outputs * 1, 1, connect_one_layer.rolling_mean, 1);
            copy_cpu_TA(layer_info->outputs, batch_parameters + layer_info->outputs * 2, 1, connect_one_layer.rolling_variance, 1);
        }
        copy_cpu_TA(layer_info->biases_length, base_parameters + layer_info->weights_length, 1, connect_one_layer.biases, 1);
        chunk_connected -= 1;
    }
    else
    {
        dechiper(base_parameters, layer_info->weights_length);
    }
    chunk_connected ++;
    copy_cpu_TA(layer_info->weights_length, base_parameters, 1, connect_one_layer.weights, 1);
    only_one_net.workspace_size = connect_one_layer.workspace_size;
    connect_one_layer.connected_index = chunk_connected - 1;
    if (chunk_connected == 1)
    {
        forward_init(connect_one_layer, only_one_net);
    }
    forward_gemm_part(connect_one_layer, only_one_net);
    if (layer_info->finish)
    {
        forward_after(connect_one_layer, only_one_net);
        if (only_one_net.workspace_size > 0) {
            free(only_one_net.workspace);
            only_one_net.workspace = NULL;
        }
        if(layer_info->batch_normalize) {
            free(connect_one_layer.scales);
            free(connect_one_layer.mean);
            free(connect_one_layer.mean_delta);
            free(connect_one_layer.variance);
            free(connect_one_layer.variance_delta);
            free(connect_one_layer.rolling_mean);
            free(connect_one_layer.rolling_variance);
            free(connect_one_layer.x);
            free(connect_one_layer.x_norm);
            connect_one_layer.scales = NULL;
            connect_one_layer.mean = NULL;
            connect_one_layer.mean_delta = NULL;
            connect_one_layer.variance = NULL;
            connect_one_layer.variance_delta = NULL;
            connect_one_layer.rolling_mean = NULL;
            connect_one_layer.rolling_variance = NULL;
            connect_one_layer.x = NULL;
            connect_one_layer.x_norm = NULL;
        }
        if (connect_one_layer.m != NULL) {
            free(connect_one_layer.m);
            free(connect_one_layer.v);
            free(connect_one_layer.bias_m);
            free(connect_one_layer.scale_m);
            free(connect_one_layer.bias_v);
            free(connect_one_layer.scale_v);
            connect_one_layer.m = NULL;
            connect_one_layer.v = NULL;
            connect_one_layer.bias_m = NULL;
            connect_one_layer.scale_m = NULL;
            connect_one_layer.bias_v = NULL;
            connect_one_layer.scale_v = NULL;
        }
        free(connect_one_layer.weights);
        connect_one_layer.weights = NULL;
        free(connect_one_layer.biases);
        connect_one_layer.biases = NULL;
        copy_output(layer_info, connect_one_layer.output, connect_one_layer.outputs);
        free(connect_one_layer.output);
        connect_one_layer.output = NULL;
        chunk_connected = 0;
    }


}
#define DECRYPT

// #define ONLY_DECRYPT
void forward_network_per_layer(float *input, float *base_parameters, float *batch_parameters, layer_info_and_weights *layer_info) {
    // layer_info->batch_normalize = 0;
    // printf("[TEE] per_laProvisionedyer start!\n");
#ifdef PRINT_TIME_DEBUG
    printf("[TEE] per_layer start!\n");
#endif
    if (layer_info->need_input > 0){
        only_one_input = input;
    }
    only_one_net.input = only_one_input;
    // return;
#ifdef DECRYPT
    if (layer_info->weights_length > 0 && layer_info->biases_length > 0) {
#ifdef PRINT_TIME
        TEE_Time start_time = { };
        TEE_Time stop_time = { };
        TEE_GetREETime(&start_time);
#endif
        // dechiper(base_parameters, layer_info->weights_length);
        // dechiper(base_parameters + layer_info->weights_length, layer_info->biases_length);
        // dechiper(base_parameters, layer_info->biases_length + layer_info->weights_length);
        weights_length_array[current_idx_weights++] = layer_info->weights_length;
        weights_length_array[current_idx_weights++] = layer_info->biases_length;
        if (layer_info->batch_normalize)
        {   
            if (layer_info->type == CONVOLUTIONAL_TA || layer_info->type == CONNECTED_TA)
            {
                weights_length_array[current_idx_weights++] = layer_info->biases_length;
                weights_length_array[current_idx_weights++] = layer_info->biases_length;
                weights_length_array[current_idx_weights++] = layer_info->biases_length;
        //         dechiper(batch_parameters + layer_info->biases_length * 0, layer_info->biases_length * 3);
        //         // dechiper(batch_parameters + layer_info->biases_length * 1, layer_info->biases_length);
        //         // dechiper(batch_parameters + layer_info->biases_length * 2, layer_info->biases_length);
            }
        }
        // aes_cbc_TA("decrypt", weights, layer_info->weights_length);
        // aes_cbc_TA("decrypt", biases, layer_info->biases_length);
#ifdef PRINT_TIME
        TEE_GetREETime(&stop_time);
        printf("[decrypt]: delta: %u(ms)\n", get_delta_time_in_ms(start_time, stop_time));
#endif
    }
#endif
// #ifdef ONLY_DECRYPT
//     return;
// #endif
    switch (layer_info->type)
    {
    case CONVOLUTIONAL_TA:
        /* code */
        run_conv_layer(layer_info, base_parameters, batch_parameters);
        break;
    case CONNECTED_TA:
        run_connected_layer(layer_info, base_parameters, batch_parameters);
        break;
    case MAXPOOL_TA:
        run_maxpool_layer(layer_info);
        break;
    case SOFTMAX_TA:
        run_softmax_layer(layer_info);
        break;
    case DROPOUT_TA:
        run_dropout_layer(layer_info);
        break;
    case AVGPOOL_TA:
        run_avgpool_layer(layer_info);
        break;
    case COST_TA:
        run_cost_layer(layer_info);
        break;
    case SHORTCUT_TA:
        run_shortcut_layer(layer_info);
        break;
    default:
        return;
        break;
    }
#ifdef PRINT_TIME_DEBUG
    printf("[TEE] per_layer finish, TYPE: %d!\n", layer_info->type);
#endif
}


void update_network_TA(update_args_TA a)
{
    int i;
    for(i = 0; i < netta.n; ++i){
        layer_TA l = netta.layers[i];
        if(l.update_TA){
            l.update_TA(l, a);
        }
    }
}


void calc_network_cost_TA()
{
    int i;
    float sum = 0;
    int count = 0;
    for(i = 0; i < netta.n; ++i){
        if(netta.layers[i].cost){
            sum += netta.layers[i].cost[0];
            ++count;
        }
    }
    *netta.cost = sum/count;
    err_sum += *netta.cost;
}


void calc_network_loss_TA(int n, int batch)
{
    float loss = (float)err_sum/(n*batch);

    if(avg_loss == -1) avg_loss = loss;
    avg_loss = avg_loss*.9 + loss*.1;

    char loss_char[20];
    char avg_loss_char[20];
    ftoa(loss, loss_char, 5);
    ftoa(avg_loss, avg_loss_char, 5);
    IMSG("loss = %s, avg loss = %s from the TA\n",loss_char, avg_loss_char);
    err_sum = 0;
}



//void backward_network_TA(float *ca_net_input, float *ca_net_delta)
void backward_network_TA(float *ca_net_input)
{
    int i;

    for(i = netta.n-1; i >= 0; --i){
        layer_TA l = netta.layers[i];

        if(l.stopbackward) break;
        if(i == 0){
            for(int z=0; z<l.inputs*l.batch; z++){
             // note: both ca_net_input and ca_net_delta are pointer
                ta_net_input[z] = ca_net_input[z];
                //ta_net_delta[z] = ca_net_delta[z]; zeros removing
                ta_net_delta[z] = 0.0f;
            }

            netta.input = ta_net_input;
            netta.delta = ta_net_delta;
        }else{
            layer_TA prev = netta.layers[i-1];
            netta.input = prev.output;
            netta.delta = prev.delta;
        }

        netta.index = i;
        l.backward_TA(l, netta);

        // when the first layer in TEE is a Dropout layer
        if((l.type == DROPOUT_TA) && (i == 0)){
            for(int z=0; z<l.inputs*l.batch; z++){
                ta_net_input[z] = l.output[z];
                ta_net_delta[z] = l.delta[z];
            }
            //netta.input = l.output;
            //netta.delta = l.delta;
        }
    }
}
