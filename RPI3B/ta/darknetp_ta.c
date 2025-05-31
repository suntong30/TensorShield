#include <tee_internal_api.h>
#include <tee_internal_api_extensions.h>

#include "darknetp_ta.h"
#include "shortcut_layer_TA.h"
#include "convolutional_layer_TA.h"
#include "maxpool_layer_TA.h"
#include "avgpool_layer_TA.h"
#include "dropout_layer_TA.h"
#include "activation_layer_TA.h"
#include "connected_layer_TA.h"
#include "softmax_layer_TA.h"
#include "cost_layer_TA.h"
#include "network_TA.h"
#include "experiment_network_TA.h"
#include "activations_TA.h"
#include "darknet_TA.h"
#include "diffprivate_TA.h"
#include "parser_TA.h"
#include "math_TA.h"
#include "batchnorm_layer_TA.h"

#define LOOKUP_SIZE 4096

float *netta_truth;
int netnum = 0;
int debug_summary_com = 0;
int debug_summary_pass = 0;
int norm_output = 1;


void summary_array(const char *print_name, float *arr, int n)
{

    float sum=0, min, max, idxzero=0;

    for(int i=0; i<n; i++)
    {
        sum = sum + arr[i];
        if (i == 0){
            min = arr[i];
            max = arr[i];
        }
        if (arr[i] < min){
            min = arr[i];
        }
        if (arr[i] > max){
            max = arr[i];
        }
        if (!(arr[i] > 0 || arr[i] < 0)){
           idxzero++;
        }
    }

    float mean=0;
    mean = sum / n;

    char mean_char[20];
    char min_char[20];
    char max_char[20];
    char idxzero_char[20];
    ftoa(mean, mean_char, 5);
    ftoa(min, min_char, 5);
    ftoa(max, max_char, 5);
    ftoa(idxzero, idxzero_char, 5);

    DMSG("%s || mean = %s; min=%s; max=%s; number of zeros=%s \n", print_name, mean_char, min_char, max_char, idxzero_char);
    (void) print_name;
}


TEE_Result TA_CreateEntryPoint(void)
{
    DMSG("has been called");

    return TEE_SUCCESS;
}

void TA_DestroyEntryPoint(void)
{
    DMSG("has been called");
}

TEE_Result TA_OpenSessionEntryPoint(uint32_t param_types,
                                    TEE_Param __maybe_unused params[4],
                                    void __maybe_unused **sess_ctx)
{
    uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE);

    DMSG("has been called");

    if (param_types != exp_param_types)
    return TEE_ERROR_BAD_PARAMETERS;

    /* Unused parameters */
    (void)&params;
    (void)&sess_ctx;

    IMSG("secure world opened!\n");
    return TEE_SUCCESS;
}


void TA_CloseSessionEntryPoint(void __maybe_unused *sess_ctx)
{
    (void)&sess_ctx; /* Unused parameter */
    IMSG("Goodbye!\n");
}

static TEE_Result make_netowork_TA_params(uint32_t param_types,
                                       TEE_Param params[4])
{
  uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                                             TEE_PARAM_TYPE_MEMREF_INPUT,
                                             TEE_PARAM_TYPE_NONE,
                                             TEE_PARAM_TYPE_NONE );

  //DMSG("has been called");
  if (param_types != exp_param_types)
  return TEE_ERROR_BAD_PARAMETERS;

    int *params0 = params[0].memref.buffer;
    float *params1 = params[1].memref.buffer;

    int n = params0[0];
    int time_steps = params0[1];
    int notruth = params0[2];
    int batch = params0[3];
    int subdivisions = params0[4];
    int random = params0[5];
    int adam = params0[6];
    int h = params0[7];
    int w = params0[8];
    int c = params0[9];
    int inputs = params0[10];
    int max_crop = params0[11];
    int min_crop = params0[12];
    int center = params0[13];
    int burn_in = params0[14];
    int max_batches = params0[15];

    float learning_rate = params1[0];
    float momentum = params1[1];
    float decay = params1[2];
    float B1 = params1[3];
    float B2 = params1[4];
    float eps = params1[5];
    float max_ratio = params1[6];
    float min_ratio = params1[7];
    float clip = params1[8];
    float angle = params1[9];
    float aspect = params1[10];
    float saturation = params1[11];
    float exposure = params1[12];
    float hue = params1[13];
    float power = params1[14];

    make_network_TA(n, learning_rate, momentum, decay, time_steps, notruth, batch, subdivisions, random, adam, B1, B2, eps, h, w, c, inputs, max_crop, min_crop, max_ratio, min_ratio, center, clip, angle, aspect, saturation, exposure, hue, burn_in, power, max_batches);

    return TEE_SUCCESS;
}

static TEE_Result update_net_agrv_TA_params(uint32_t param_types,
                                       TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_VALUE_INPUT,
                                               TEE_PARAM_TYPE_MEMREF_INOUT,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE);

    //DMSG("has been called");
    if (param_types != exp_param_types)
    return TEE_ERROR_BAD_PARAMETERS;

    netta.workspace = params[1].memref.buffer;

    return TEE_SUCCESS;
}


static TEE_Result make_activation_layer_TA_params(uint32_t param_types,
                                       TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                                            TEE_PARAM_TYPE_VALUE_INPUT,
                                            TEE_PARAM_TYPE_MEMREF_INPUT,
                                            TEE_PARAM_TYPE_NONE);
    if (param_types != exp_param_types)
        return TEE_ERROR_BAD_PARAMETERS;
    int *params0 = params[0].memref.buffer;
    char *params1 = params[1].memref.buffer;
    char *acti = params1;
    ACTIVATION_TA activation = get_activation_TA(acti);
    int batch = params0[0];
    int inputs = params0[1];
    int current_layer_n = params0[2];
    layer_TA lta = make_activation_layer_TA(batch, inputs, activation);
    if (lta.workspace_size > netta.workspace_size) netta.workspace_size = lta.workspace_size;
    lta.netnum = current_layer_n;
    netta.layers[netnum] = lta;
    netnum++;
    return TEE_SUCCESS;

}
static TEE_Result make_maxpool_layer_TA_params(uint32_t param_types,
                                       TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE);

    //DMSG("has been called");
    if (param_types != exp_param_types)
    return TEE_ERROR_BAD_PARAMETERS;

    int *params0 = params[0].memref.buffer;

    int batch = params0[0];
    int h = params0[1];
    int w = params0[2];
    int c = params0[3];
    int size = params0[4];
    int stride = params0[5];
    int padding = params0[6];
    int current_layer_n = params0[7];
    layer_TA lta = make_maxpool_layer_TA(batch, h, w, c, size, stride, padding);
    lta.netnum = current_layer_n;
    netta.layers[netnum] = lta;
    netnum++;

    return TEE_SUCCESS;
}

static TEE_Result make_batchnorm_layer_TA_params(uint32_t param_types,
                                       TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE);

    //DMSG("has been called");
    if (param_types != exp_param_types)
    return TEE_ERROR_BAD_PARAMETERS;

    int *params0 = params[0].memref.buffer;

    int batch = params0[0];
    int h = params0[1];
    int w = params0[2];
    int c = params0[3];
    int current_layer_n = params0[4];
    layer_TA lta = make_batchnorm_layer_TA(batch, w, h, c);
    lta.netnum = current_layer_n;
    netta.layers[netnum] = lta;
    netnum++;

    return TEE_SUCCESS;
}

static TEE_Result make_convolutional_layer_TA_params(uint32_t param_types,
                                       TEE_Param params[4])
{
  uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                                             TEE_PARAM_TYPE_VALUE_INPUT,
                                             TEE_PARAM_TYPE_MEMREF_INPUT,
                                             TEE_PARAM_TYPE_NONE);

  //DMSG("has been called");
  if (param_types != exp_param_types)
  return TEE_ERROR_BAD_PARAMETERS;

    int *params0 = params[0].memref.buffer;
    float params1 = params[1].value.a;
    char *params2 = params[2].memref.buffer;

    int batch = params0[0];
    int h = params0[1];
    int w = params0[2];
    int c = params0[3];
    int n = params0[4];
    int groups = params0[5];
    int size = params0[6];
    int stride = params0[7];
    int padding = params0[8];
    int batch_normalize = params0[9];
    int binary = params0[10];
    int xnor = params0[11];
    int adam = params0[12];
    int flipped = params0[13];
    int current_layer_n = params0[14];
    float dot = params1;
    char *acti = params2;

    ACTIVATION_TA activation = get_activation_TA(acti);

    layer_TA lta = make_convolutional_layer_TA_new(batch, h, w, c, n, groups, size, stride, padding, activation, batch_normalize, binary, xnor, adam, flipped, dot);
    lta.netnum = current_layer_n;
    netta.layers[netnum] = lta;
    
    if (lta.workspace_size > netta.workspace_size) netta.workspace_size = lta.workspace_size;
    netnum++;

    return TEE_SUCCESS;
}

// layer_TA make_shortcut_layer_TA(int batch, int index, int w, int h, int c, int w2, int h2, int c2);

static TEE_Result make_shortcut_layer_TA_params(uint32_t param_types,
                                       TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE);

    //DMSG("has been called");
    if (param_types != exp_param_types)
    return TEE_ERROR_BAD_PARAMETERS;

    int *params0 = params[0].memref.buffer;

    int batch = params0[0];
    int index = params0[1];
    int w = params0[2];
    int h = params0[3];
    int c = params0[4];
    int w2 = params0[5];
    int h2 = params0[6];
    int c2 = params0[7];
    int current_layer_n = params0[8];
    layer_TA lta = make_shortcut_layer_TA(batch, index, w, h, c, w2, h2, c2);
    lta.netnum = current_layer_n;
    netta.layers[netnum] = lta;
    netnum++;

    return TEE_SUCCESS;
}

static TEE_Result make_avgpool_layer_TA_params(uint32_t param_types,
                                       TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE);

    //DMSG("has been called");
    if (param_types != exp_param_types)
    return TEE_ERROR_BAD_PARAMETERS;

    int *params0 = params[0].memref.buffer;

    int batch = params0[0];
    int h = params0[1];
    int w = params0[2];
    int c = params0[3];
    int current_layer_n = params0[4];
    layer_TA lta = make_avgpool_layer_TA(batch, h, w, c);
    lta.netnum = current_layer_n;
    netta.layers[netnum] = lta;
    netnum++;

    return TEE_SUCCESS;
}

static TEE_Result make_dropout_layer_TA_params(uint32_t param_types,
                                       TEE_Param params[4])
{
  uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                                             TEE_PARAM_TYPE_MEMREF_INPUT,
                                             TEE_PARAM_TYPE_MEMREF_INPUT,
                                             TEE_PARAM_TYPE_MEMREF_INPUT);

  //DMSG("has been called");
  if (param_types != exp_param_types)
  return TEE_ERROR_BAD_PARAMETERS;

    int *params0 = params[0].memref.buffer;
    float *params1 = params[1].memref.buffer;
    float *params2 = params[2].memref.buffer;
    float *params3 = params[3].memref.buffer;
    int buffersize = params[2].memref.size / sizeof(float);

    int *passint;
    passint = params0;
    int batch = passint[0];
    int inputs = passint[1];
    int w = passint[2];
    int h = passint[3];
    int c = passint[4];
    int current_layer_n = passint[5];
    float probability = params1[0];

    float *net_prev_output = params2;
    float *net_prev_delta = params3;

    layer_TA lta = make_dropout_layer_TA_new(batch, inputs, probability, w, h, c, netnum);
    lta.netnum = current_layer_n;

    if(netnum == 0){
      for(int z=0; z<buffersize; z++){
        lta.output[z] = net_prev_output[z];
        lta.delta[z] = net_prev_delta[z];
      }
    }else{
        lta.output = netta.layers[netnum-1].output;
        lta.delta = netta.layers[netnum-1].delta;
    }

    netta.layers[netnum] = lta;
    netnum++;

    return TEE_SUCCESS;
}


static TEE_Result make_connected_layer_TA_params(uint32_t param_types,
                                       TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE);

    //DMSG("has been called");

    if (param_types != exp_param_types)
    return TEE_ERROR_BAD_PARAMETERS;

    int *passarg;
    passarg = params[0].memref.buffer;
    int batch = passarg[0];
    int inputs = passarg[1];
    int outputs = passarg[2];
    int batch_normalize = passarg[3];
    int adam = passarg[4];
    int current_layer_n = passarg[5];
    char *acti;
    acti = params[1].memref.buffer;
    ACTIVATION_TA activation = get_activation_TA(acti);

    layer_TA lta = make_connected_layer_TA_new(batch, inputs, outputs, activation, batch_normalize, adam);
    lta.netnum = current_layer_n;

    netta.layers[netnum] = lta;
    netnum++;

    return TEE_SUCCESS;
}

static TEE_Result make_softmax_layer_TA_params(uint32_t param_types,
                                       TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_VALUE_INPUT,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE);

    //DMSG("has been called");

    if (param_types != exp_param_types)
    return TEE_ERROR_BAD_PARAMETERS;

    int *params0 = params[0].memref.buffer;
    int batch = params0[0];
    int inputs = params0[1];
    int groups = params0[2];
    int w = params0[3];
    int h = params0[4];
    int c = params0[5];
    int spatial = params0[6];
    int noloss = params0[7];
    float temperature = params[1].value.a;
    int current_layer_n = params0[8];
    layer_TA lta = make_softmax_layer_TA_new(batch, inputs, groups, temperature, w, h, c, spatial, noloss);
    lta.netnum = current_layer_n;
    netta.layers[netnum] = lta;
    netnum++;

    return TEE_SUCCESS;
}

static TEE_Result make_cost_layer_TA_params(uint32_t param_types,
                                       TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_NONE);

    //DMSG("has been called");

    if (param_types != exp_param_types)
    return TEE_ERROR_BAD_PARAMETERS;

    int *params0 = params[0].memref.buffer;
    int batch = params0[0];
    int inputs = params0[1];
    int current_layer_n = params0[2];
    float *params1 = params[1].memref.buffer;
    float scale = params1[0];
    float ratio = params1[1];
    float noobject_scale = params1[2];
    float thresh = params1[3];

    char *cost_t;
    cost_t = params[2].memref.buffer;
    ACTIVATION_TA cost_type = get_cost_type_TA(cost_t);


    layer_TA lta = make_cost_layer_TA_new(batch, inputs, cost_type, scale, ratio, noobject_scale, thresh);
    lta.netnum = current_layer_n;
    netta.layers[netnum] = lta;
    netnum++;

    // allocate net.truth when the cost layer inside TEE
    netta_truth = malloc(inputs * batch * sizeof(float));
    //free(netta_truth) needed

    return TEE_SUCCESS;
}


static TEE_Result transfer_weights_TA_params(uint32_t param_types,
                                             TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_VALUE_INPUT,
                                               TEE_PARAM_TYPE_NONE);

    //DMSG("has been called");
    // IMSG("test1: param_type: %u, exp_param_types: %u\n", param_types, exp_param_types);
    if (param_types != exp_param_types)
        return TEE_ERROR_BAD_PARAMETERS;
    // IMSG("test2\n");

    float *vec = params[0].memref.buffer;

    int *params1 = params[1].memref.buffer;
    int length = params1[0];
    int layer_i = params1[1];
    int additional = params1[2];
    // for (int i = 0; i < netta.n; ++i)
    // {
    //     printf("[TEE] layer %d: %d\n", i, netta.layers[i].type);
    // }
    char type = params[2].value.a;
    // printf("[TEE] transfer %d %d\n", length, layer_i);
    load_weights_TA(vec, length, layer_i, type, additional);

    return TEE_SUCCESS;
}

static TEE_Result save_weights_TA_params(uint32_t param_types,
                                             TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_OUTPUT,
                                               TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_VALUE_INPUT,
                                               TEE_PARAM_TYPE_NONE);

    //DMSG("has been called");

    if (param_types != exp_param_types)
        return TEE_ERROR_BAD_PARAMETERS;

    float *vec = params[0].memref.buffer;

    int *params1 = params[1].memref.buffer;
    int length = params1[0];
    int layer_i = params1[1];

    char type = params[2].value.a;

    float *weights_encrypted = malloc(sizeof(float)*length);
    save_weights_TA(weights_encrypted, length, layer_i, type);

    for(int z=0; z<length; z++){
        vec[z] = weights_encrypted[z];
    }

    free(weights_encrypted);
    return TEE_SUCCESS;
}



static TEE_Result forward_network_TA_params(uint32_t param_types,
                                          TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES( TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_VALUE_INPUT,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE);
    //TEE_PARAM_TYPE_VALUE_INPUT

    //DMSG("has been called");

    if (param_types != exp_param_types)
    return TEE_ERROR_BAD_PARAMETERS;

    float *net_input = params[0].memref.buffer;
    int net_train = params[1].value.a;

    netta.input = net_input;
    netta.train = net_train;

    if(debug_summary_com == 1){
        summary_array("forward_network / net.input", netta.input, params[0].memref.size / sizeof(float));
    }
    forward_network_TA();

    return TEE_SUCCESS;
}

static TEE_Result first_forward_network_part_TA_params(uint32_t param_types,
                                          TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES( TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE);
    if (param_types != exp_param_types)
        return TEE_ERROR_BAD_PARAMETERS;
    first_network();
    return TEE_SUCCESS;
}
static TEE_Result free_forward_network_part_TA_params(uint32_t param_types,
                                          TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES( TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE);
    if (param_types != exp_param_types)
        return TEE_ERROR_BAD_PARAMETERS;
    free_network();
    return TEE_SUCCESS;
}

static TEE_Result forward_network_part_TA_params(uint32_t param_types,
                                          TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES( TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_VALUE_INPUT,
                                               TEE_PARAM_TYPE_VALUE_INPUT,
                                               TEE_PARAM_TYPE_MEMREF_OUTPUT);
    if (param_types != exp_param_types)
    return TEE_ERROR_BAD_PARAMETERS;

    float *net_input = params[0].memref.buffer;
    int net_train = params[1].value.a;
    // int layer_forward_start_idx = params[1].value.b;
    int start_idx = params[2].value.a;
    int end_idx = params[2].value.b;
    netta.input = net_input;
    netta.train = net_train;
    // int res = forward_network_part_TA(layer_forward_start_idx);
    int res = forward_network_part_TA(start_idx, end_idx);

    float *params3 = params[3].memref.buffer;
    int buffersize = params[3].memref.size / sizeof(float);
    for(int z=0; z<buffersize; z++){
        params3[z] = netta.layers[netta.n-1].output[z];
    }
    // params[1].value.b = res;
    // // free(input);

    return TEE_SUCCESS;
}

static TEE_Result tt_forward_network_TA_params(uint32_t param_types,
                                          TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES( TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_VALUE_INPUT,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE);

    if (param_types != exp_param_types)
    return TEE_ERROR_BAD_PARAMETERS;

    float *net_input = params[0].memref.buffer;
    int net_train = params[1].value.a;
    int n_layer_th = params[1].value.b;


    netta.input = net_input;
    netta.train = net_train;

    if(debug_summary_com == 1){
        summary_array("forward_network / net.input", netta.input, params[0].memref.size / sizeof(float));
    }
    tt_forward_network_TA(n_layer_th);

    return TEE_SUCCESS;
}


static TEE_Result forward_network_TA_params_DEFUSION(uint32_t param_types,
                                          TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES( TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_VALUE_INPUT,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE);
    //TEE_PARAM_TYPE_VALUE_INPUT

    //DMSG("has been called");

    if (param_types != exp_param_types)
    return TEE_ERROR_BAD_PARAMETERS;

    float *net_input = params[0].memref.buffer;
    int net_train = params[1].value.a;

    netta.input = net_input;
    netta.train = net_train;

    // if(debug_summary_com == 1){
    //     summary_array("forward_network / net.input", netta.input, params[0].memref.size / sizeof(float));
    // }
    // forward_network_TA();
    int channel = params[1].value.a;
    int size = params[1].value.b;
    forward_network_TA_DEDefusion(channel, size);

    write_mask_test(net_input, size * size *channel);

    return TEE_SUCCESS;
}

//
// static TEE_Result forward_network_TA_params(uint32_t param_types,
//                                           TEE_Param params[4])
// {
//     uint32_t exp_param_types = TEE_PARAM_TYPES( TEE_PARAM_TYPE_MEMREF_INPUT,
//                                                TEE_PARAM_TYPE_VALUE_INPUT,
//                                                TEE_PARAM_TYPE_NONE,
//                                                TEE_PARAM_TYPE_NONE);
//     //TEE_PARAM_TYPE_VALUE_INPUT
//
//     //DMSG("has been called");
//
//     if (param_types != exp_param_types)
//     return TEE_ERROR_BAD_PARAMETERS;
//
//     float *net_input = params[0].memref.buffer;
//     int net_train = params[1].value.a;
//
//     netta.input = net_input;
//     netta.train = net_train;
//
//     forward_network_TA();
//
//     return TEE_SUCCESS;
// }


static TEE_Result forward_network_back_TA_params(uint32_t param_types,
                                           TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_OUTPUT,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE);
    if (param_types != exp_param_types)
        return TEE_ERROR_BAD_PARAMETERS;

    float *params0 = params[0].memref.buffer;
    int buffersize = params[0].memref.size / sizeof(float);
    // for(int z=0; z<buffersize; z++){
    //     params0[z] = netta.layers[netta.n-1].output[z];
    // }
    for (int z=0; z<buffersize; z++)
    {
        params0[z] = only_one_input[z];
    }
    // ?????
    //free(ta_net_input);
    free(only_one_input);
    if(debug_summary_com == 1){
        summary_array("forward_network_back / l_pp2.output", netta.layers[netta.n-1].output, buffersize);
    }
    return TEE_SUCCESS;
}

static TEE_Result tt_init_mask_params(uint32_t param_types,
                                       TEE_Param params[4])
{
    IMSG("tt_init_mask_params\n");
    uint32_t exp_param_types = TEE_PARAM_TYPES( TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_VALUE_INPUT,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE);
    if (param_types != exp_param_types)
        return TEE_ERROR_BAD_PARAMETERS;
    IMSG("tt_init_mask_params111\n");

    float *params0 = params[0].memref.buffer;
    int size = params[0].memref.size / sizeof(float);
    IMSG("tt_init_mask_params_2\n");
    // float *params0 = malloc(sizeof(float) * size);
    write_mask_test(params0, size);
    // free(params0);
    return TEE_SUCCESS;
}

static TEE_Result tt_free_params(uint32_t param_types,
                                       TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES( TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE);
    if (param_types != exp_param_types)
        return TEE_ERROR_BAD_PARAMETERS;

    float *params0 = params[0].memref.buffer;
    int size = params[0].memref.size / sizeof(float);
    // float *params0 = malloc(sizeof(float) * size);
    free_simulate_arry();
    // free(params0);
    return TEE_SUCCESS;
}

static TEE_Result tt_forward_relu_params(uint32_t param_types,
                                       TEE_Param params[4])
{
    // IMSG("relu test\n");
    uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INOUT,
                                            TEE_PARAM_TYPE_VALUE_INPUT,
                                            TEE_PARAM_TYPE_VALUE_INOUT,
                                            TEE_PARAM_TYPE_VALUE_INOUT);
    // IMSG("%u,%u\n", param_types, exp_param_types);
    
    if (param_types != exp_param_types)
    {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    float *params0 = params[0].memref.buffer;
    int size = params[0].memref.size / sizeof(float);
    int activation = params[1].value.a;
    int out_channels = params[1].value.b;
    int size_y = params[2].value.a;
    int use_mask = params[2].value.b;
    int flops = params[3].value.a;
    // float *output = malloc(sizeof(float) * size);
    // for (int i = 0; i < size; ++i){
    //     output[i] = params0[i];
    // }
    float *output;
    if (size > 112 * 112 * 128)
    {
        output = params0;
    }
    else 
    {
        output = malloc(sizeof(float) * size);
        for (int i = 0; i < size; ++i)
        {
            output[i] = params0[i];
        }
    }
    tt_forward_relu(output, size, activation, out_channels, size_y, use_mask, flops);
    for (int i = 0; i < size; ++i) {
        params0[i] = output[i];
    }
    params[2].value.a = deobf_time_ms_flops;
    params[2].value.b = relu_time_ms_flops;
    params[3].value.a = mask_time_ms_flops;
    if (size <= 112 * 112 * 128)
    {
        free(output);
    }
    return TEE_SUCCESS;

}

float *base_parameter_whole;


// #define DEBUG
static TEE_Result forward_network_per_layer_params(uint32_t param_types,
                                           TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES( TEE_PARAM_TYPE_MEMREF_INPUT,
                                            TEE_PARAM_TYPE_MEMREF_INPUT,
                                            TEE_PARAM_TYPE_MEMREF_INPUT,
                                            TEE_PARAM_TYPE_MEMREF_INPUT);
    if (param_types != exp_param_types)
        return TEE_ERROR_BAD_PARAMETERS;
    float *input;
    size_t input_size = 0;
    float *base_parameters = params[1].memref.buffer;
    float *batch_parameters = params[2].memref.buffer;
    layer_info_and_weights *layer_info = params[3].memref.buffer;
    if (layer_info->need_input > 0){
        input = params[0].memref.buffer;
        input_size = params[0].memref.size / sizeof(float);
    }
#ifdef PRINT_TIME_DEBUG
    printf("[REE] type: %d, activation: %d, batch: %d, h: %d\n",
        layer_info->type, layer_info->activation, layer_info->batch, layer_info->h);
    printf("[REE] w: %d, c: %d, n: %d, groups: %d\n",
        layer_info->w, layer_info->c, layer_info->n, layer_info->groups);
    printf("[REE] size: %d, stride: %d, padding: %d, batch_normalize: %d\n",
        layer_info->size, layer_info->stride, layer_info->padding, layer_info->batch_normalize);
    printf("[REE] xnor: %d, binary: %d, adam: %d, flipped: %d\n",
        layer_info->xnor, layer_info->binary, layer_info->adam, layer_info->flipped);
    printf("[REE] cost_type: %d, biases_length: %d, weights_length: %d, inputs: %d\n",
        layer_info->cost_type, layer_info->biases_length, layer_info->weights_length, layer_info->inputs);
    printf("[REE] outputs: %d\n",
        layer_info->outputs);   
    printf("[REE] layer_info: %dB, input_size: %dB, parameter: %d, %dB\n", params[3].memref.size, 
        params[0].memref.size, params[1].memref.size, params[2].memref.size);
    printf("[REE] layer:%d\n", layer_info->need_input);
#endif
    forward_network_per_layer(input, base_parameters, batch_parameters, layer_info);
    return TEE_SUCCESS;
}

static TEE_Result forward_connect_layer_part_params(uint32_t param_types,
                                           TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES( TEE_PARAM_TYPE_MEMREF_INPUT,
                                            TEE_PARAM_TYPE_MEMREF_INPUT,
                                            TEE_PARAM_TYPE_MEMREF_INPUT,
                                            TEE_PARAM_TYPE_MEMREF_INPUT);
    if (param_types != exp_param_types)
        return TEE_ERROR_BAD_PARAMETERS;
    float *input;
    size_t input_size = 0;
    float *base_parameters = params[1].memref.buffer;
    float *batch_parameters = params[2].memref.buffer;
    layer_info_and_weights *layer_info = params[3].memref.buffer;
    if (layer_info->need_input > 0){
        input = params[0].memref.buffer;
        input_size = params[0].memref.size / sizeof(float);
    }
#ifdef PRINT_TIME_DEBUG
    printf("[REE] type: %d, activation: %d, batch: %d, h: %d\n",
        layer_info->type, layer_info->activation, layer_info->batch, layer_info->h);
    printf("[REE] w: %d, c: %d, n: %d, groups: %d\n",
        layer_info->w, layer_info->c, layer_info->n, layer_info->groups);
    printf("[REE] size: %d, stride: %d, padding: %d, batch_normalize: %d\n",
        layer_info->size, layer_info->stride, layer_info->padding, layer_info->batch_normalize);
    printf("[REE] xnor: %d, binary: %d, adam: %d, flipped: %d\n",
        layer_info->xnor, layer_info->binary, layer_info->adam, layer_info->flipped);
    printf("[REE] cost_type: %d, biases_length: %d, weights_length: %d, inputs: %d\n",
        layer_info->cost_type, layer_info->biases_length, layer_info->weights_length, layer_info->inputs);
    printf("[REE] outputs: %d\n",
        layer_info->outputs);   
    printf("[REE] layer_info: %dB, input_size: %dB, parameter: %d, %dB\n", params[3].memref.size, 
        params[0].memref.size, params[1].memref.size, params[2].memref.size);
    printf("[REE] layer:%d\n", layer_info->need_input);
#endif
    run_connected_layer_part(layer_info, base_parameters, batch_parameters);
    return TEE_SUCCESS;
}


static TEE_Result get_back_run_time_ms(uint32_t param_types,
                                           TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES( TEE_PARAM_TYPE_VALUE_OUTPUT,
                                            TEE_PARAM_TYPE_VALUE_OUTPUT,
                                            TEE_PARAM_TYPE_NONE,
                                            TEE_PARAM_TYPE_NONE);
    if (param_types != exp_param_types)
        return TEE_ERROR_BAD_PARAMETERS;
    params[0].value.a = deobf_time_ms;
    params[0].value.b = mask_time_ms;
    params[1].value.a = tee_operator_ms;
    return TEE_SUCCESS;
}

static TEE_Result tt_forward_network_back_one_layer_TA_params(uint32_t param_types,
                                           TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES( TEE_PARAM_TYPE_MEMREF_OUTPUT,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE);
    if (param_types != exp_param_types)
        return TEE_ERROR_BAD_PARAMETERS;

    float *params0 = params[0].memref.buffer;
    int buffersize = params[0].memref.size / sizeof(float);
    int n_layer_th = params[1].value.a;

    for(int z=0; z<buffersize; z++){
        params0[z] = netta.layers[n_layer_th].output[z];
    }
    return TEE_SUCCESS;
}



static TEE_Result backward_network_TA_params(uint32_t param_types,
                                           TEE_Param params[4])
{


    uint32_t exp_param_types = TEE_PARAM_TYPES( TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_VALUE_INPUT,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE);
    //TEE_PARAM_TYPE_VALUE_INPUT

    //DMSG("has been called");

    if (param_types != exp_param_types)
    return TEE_ERROR_BAD_PARAMETERS;

    float *params0 = params[0].memref.buffer;
    //float *params1 = params[1].memref.buffer;
    int net_train = params[1].value.a;

    netta.train = net_train;

    if(debug_summary_com == 1){
        summary_array("backward_network / l_pp1.output", params0, params[0].memref.size / sizeof(float));
        //summary_array("backward_network / l_pp1.delta", params1, params[1].memref.size / sizeof(float));
    }
    //backward_network_TA(params0, params1); //zeros, removing
    backward_network_TA(params0);

    return TEE_SUCCESS;
}



static TEE_Result backward_network_TA_addidion_params(uint32_t param_types,
                                           TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES( TEE_PARAM_TYPE_MEMREF_OUTPUT,
                                               TEE_PARAM_TYPE_MEMREF_OUTPUT,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE);
    if (param_types != exp_param_types)
        return TEE_ERROR_BAD_PARAMETERS;
    //float *ltaoutput_diff = diff_private(lta.output, lta.outputs*lta.batch, 4.0f, 4.0f);
    //float *ltadelta_diff = diff_private(lta.delta, lta.outputs*lta.batch, 4.0f, 4.0f);
    //IMSG("diff");


    float *params0 = params[0].memref.buffer;
    float *params1 = params[1].memref.buffer;
    int buffersize = params[0].memref.size / sizeof(float);
    for(int z=0; z<buffersize; z++){
        params0[z] = ta_net_input[z];
        params1[z] = ta_net_delta[z];
    }
    //free(ta_net_input);
    //free(ta_net_delta);
    //free(ltaoutput_diff);
    //free(ltadelta_diff);

    if(debug_summary_com == 1){
        summary_array("backward_network_addidion / l_pp1.output", ta_net_input, buffersize);
        summary_array("backward_network_addidion / l_pp1.delta", ta_net_delta, buffersize);
    }
    return TEE_SUCCESS;
}


static TEE_Result backward_network_back_TA_params(uint32_t param_types,
                                           TEE_Param params[4])
{


    uint32_t exp_param_types = TEE_PARAM_TYPES( TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE);
    //TEE_PARAM_TYPE_VALUE_INPUT

    //DMSG("has been called");
    if (param_types != exp_param_types)
        return TEE_ERROR_BAD_PARAMETERS;

    float *params0 = params[0].memref.buffer;
    float *params1 = params[1].memref.buffer;
    int buffersize = params[0].memref.size / sizeof(float);

    for(int z=0; z<buffersize; z++){
        netta.layers[netta.n - 1].output[z] = params0[z];
        netta.layers[netta.n - 1].delta[z] = params1[z];
    }

#ifdef DEBUG
    IMSG("[TEE] backward_network_back_TA_params:");
    for(int z=0; z<10; z++){
        IMSG("%f:%f ", params0[z], params1[z]);
    }
#endif

    if(debug_summary_com == 1){
        summary_array("backward_network_back / l_pp2.output", netta.layers[netta.n - 1].output, buffersize);
        summary_array("backward_network_back / l_pp2.delta", netta.layers[netta.n - 1].delta, buffersize);
    }

    return TEE_SUCCESS;
}



static TEE_Result backward_network_back_TA_addidion_params(uint32_t param_types,
                                           TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES( TEE_PARAM_TYPE_MEMREF_OUTPUT,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE);
    if (param_types != exp_param_types)
        return TEE_ERROR_BAD_PARAMETERS;

    float *params0 = params[0].memref.buffer;
    //float *params1 = params[1].memref.buffer;
    int buffersize = params[0].memref.size / sizeof(float);

    for(int z=0; z<buffersize; z++){
        params0[z] = netta.layers[netta.n - 1].output[z];
        //params1[z] = netta.layers[netta.n - 1].delta[z]; zeros, removing
    }

    if(debug_summary_com == 1){
        summary_array("backward_network_back_addidion / l_pp2.output", netta.layers[netta.n - 1].output, buffersize);
        //summary_array("backward_network_back_addidion / l_pp2.delta", netta.layers[netta.n - 1].delta, buffersize);
    }
    return TEE_SUCCESS;
}
//
// static TEE_Result backward_network_back_TA_params(uint32_t param_types,
//                                            TEE_Param params[4])
// {
//
//
//     uint32_t exp_param_types = TEE_PARAM_TYPES( TEE_PARAM_TYPE_MEMREF_INPUT,
//                                                TEE_PARAM_TYPE_MEMREF_INPUT,
//                                                TEE_PARAM_TYPE_VALUE_INPUT,
//                                                TEE_PARAM_TYPE_NONE);
//     //TEE_PARAM_TYPE_VALUE_INPUT
//
//     //DMSG("has been called");
//
//     if (param_types != exp_param_types)
//     return TEE_ERROR_BAD_PARAMETERS;
//
//     float *ca_net_input = params[0].memref.buffer;
//     float *ca_net_delta = params[1].memref.buffer;
//     int net_train = params[2].value.a;
//
//     netta.train = net_train;
//
//     backward_network_TA(ca_net_input, ca_net_delta);
//
//     return TEE_SUCCESS;
// }

static TEE_Result update_network_TA_params(uint32_t param_types,
                                         TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES( TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE);
    //TEE_PARAM_TYPE_VALUE_INPUT

    //DMSG("has been called");

    if (param_types != exp_param_types)
    return TEE_ERROR_BAD_PARAMETERS;

    int *params0 = params[0].memref.buffer;
    float *params1 = params[1].memref.buffer;

    update_args_TA a;
    a.batch = params0[0];
    a.adam = params0[1];
    a.t = params0[2];
    a.learning_rate = params1[0];
    a.momentum = params1[1];
    a.decay = params1[2];
    a.B1 = params1[3];
    a.B2 = params1[4];
    a.eps = params1[5];

    update_network_TA(a);
    mdbg_check(1);

    return TEE_SUCCESS;
}

static TEE_Result net_truth_TA_params(uint32_t param_types,
                                         TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES( TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE);
    //TEE_PARAM_TYPE_VALUE_INPUT

    //DMSG("has been called");

    if (param_types != exp_param_types)
    return TEE_ERROR_BAD_PARAMETERS;

    int size_truth = params[0].memref.size;
    float *params0 = params[0].memref.buffer;

    for(int z=0; z<(int)(size_truth/sizeof(float)); z++){
        netta_truth[z] = params0[z];
    }
    netta.truth = netta_truth;

    return TEE_SUCCESS;
}

static TEE_Result calc_network_loss_TA_params(uint32_t param_types,
                                         TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES( TEE_PARAM_TYPE_MEMREF_INPUT,
                                             TEE_PARAM_TYPE_NONE,
                                             TEE_PARAM_TYPE_NONE,
                                             TEE_PARAM_TYPE_NONE);
    (void) exp_param_types;
    int *params0 = params[0].memref.buffer;
    int n = params0[0];
    int batch = params0[1];

    calc_network_loss_TA(n, batch);

    return TEE_SUCCESS;
}


static TEE_Result net_output_return_TA_params(uint32_t param_types,
                                              TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES( TEE_PARAM_TYPE_MEMREF_OUTPUT,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE);

    if (param_types != exp_param_types)
        return TEE_ERROR_BAD_PARAMETERS;

    float *params0 = params[0].memref.buffer;
    int buffersize = params[0].memref.size / sizeof(float);

    if(norm_output){
        // remove confidence scores
        float maxconf; maxconf = 0.00001f;
        int maxidx; maxidx = 0;

        for(int z=0; z<buffersize; z++){
            if(ta_net_output[z] > maxconf){
                maxconf = ta_net_output[z];
                maxidx = z;
            }
            ta_net_output[z] = 0.0f;
        }
        ta_net_output[maxidx] = 1.00f;
    }

    for(int z=0; z<buffersize; z++){
        params0[z] = ta_net_output[z];
    }

    free(ta_net_output);

    return TEE_SUCCESS;

}

TEE_Result TA_InvokeCommandEntryPoint(void __maybe_unused *sess_ctx,
                                      uint32_t cmd_id,
                                      uint32_t param_types, TEE_Param params[4])
{
    (void)&sess_ctx; /* Unused parameter */

    switch (cmd_id) {
        case MAKE_NETWORK_CMD:
        return make_netowork_TA_params(param_types, params);

        case WORKSPACE_NETWORK_CMD:
        return update_net_agrv_TA_params(param_types, params);

        case MAKE_CONV_CMD:
        return make_convolutional_layer_TA_params(param_types, params);

        case MAKE_ACTIVATE_CMD:
        return make_activation_layer_TA_params(param_types, params);
        case MAKE_MAX_CMD:
        return make_maxpool_layer_TA_params(param_types, params);

        case MAKE_AVG_CMD:
        return make_avgpool_layer_TA_params(param_types, params);

        case MAKE_DROP_CMD:
        return make_dropout_layer_TA_params(param_types, params);

        case MAKE_CONNECTED_CMD:
        return make_connected_layer_TA_params(param_types, params);
        case MAKE_BATCHNORM_CMD:
        return make_batchnorm_layer_TA_params(param_types, params);
        case MAKE_SOFTMAX_CMD:
        return make_softmax_layer_TA_params(param_types, params);

        case MAKE_COST_CMD:
        return make_cost_layer_TA_params(param_types, params);

        case TRANS_WEI_CMD:
        return transfer_weights_TA_params(param_types, params);

        case SAVE_WEI_CMD:
            return save_weights_TA_params(param_types, params);

        case FORWARD_CMD:
        return forward_network_TA_params(param_types, params);

        case FUSION_TEST:
        return forward_network_TA_params_DEFUSION(param_types, params);
        
        case BACKWARD_CMD:
        return backward_network_TA_params(param_types, params);

        case BACKWARD_ADD_CMD:
        return backward_network_TA_addidion_params(param_types, params);

        case UPDATE_CMD:
        return update_network_TA_params(param_types, params);

        case NET_TRUTH_CMD:
        return net_truth_TA_params(param_types, params);

        case CALC_LOSS_CMD:
        return calc_network_loss_TA_params(param_types, params);

        case OUTPUT_RETURN_CMD:
        return net_output_return_TA_params(param_types, params);

        case FORWARD_BACK_CMD:
        return forward_network_back_TA_params(param_types, params);

        case BACKWARD_BACK_CMD:
        return backward_network_back_TA_params(param_types, params);

        case BACKWARD_BACK_ADD_CMD:
        return backward_network_back_TA_addidion_params(param_types, params);

        case FORWARD_CMD_NO_CONV_TEST:
        return tt_forward_network_TA_params(param_types, params);

        case FORWARD_BACK_CMD_ONE_LAYER:
        return tt_forward_network_back_one_layer_TA_params(param_types, params);

        case FORWARD_CMD_RELU:
        return tt_forward_relu_params(param_types, params);
        case INIT_CMD_MASK:
        return tt_init_mask_params(param_types, params);
        case FREE_CMD:
        return tt_free_params(param_types, params);
        case FORWARD_CMD_PER_LAYER:
        return forward_network_per_layer_params(param_types, params);
        case FORWARD_CMD_CONNECT_LAYER_PART:
        return forward_connect_layer_part_params(param_types, params);
        case FORWARD_CMD_PART:
        return forward_network_part_TA_params(param_types, params);
        case MAKE_SHORTCUT_LAYER_CMD:
        return make_shortcut_layer_TA_params(param_types, params);
        case FIRST_N_CMD:
        return first_forward_network_part_TA_params(param_types, params);
        case FREE_N_CMD:
        return free_forward_network_part_TA_params(param_types, params);
        case RETURN_RUN_TIME_CMD:
        return get_back_run_time_ms(param_types, params);
        default:
        return TEE_ERROR_BAD_PARAMETERS;
    }
}
