#include <err.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

#include "darknet.h"
#include "activations.h"
#include "cost_layer.h"

#include "main.h"

/* OP-TEE TEE client API (built by optee_client) */
#include <tee_client_api.h>

/* TEE resources */
TEEC_Context ctx;
TEEC_Session sess;
TEEC_SharedMemory workspaceSM;

float *net_input_back;
float *net_delta_back;
float *net_output_back;
int sysCount = 0;
char state;
int debug_plot_bool = 0;
int test_test_value = 0;

void debug_plot(char *filename, int num, float *tobeplot, int length)
{
    struct stat st = {0};
    if (stat("/media/debug_plot", &st) == -1) {
            mkdir("/media/debug_plot", 0700);
    }

    FILE * fp;
    int i;

    char strnum[10];
    sprintf(strnum, "%d", num);

    /* open the file for writing*/
    char *s1 = "/media/debug_plot/";
    //char *s1 = "";
    char *s2 = ".txt";

    char *result = malloc(strlen(s1) + strlen(filename) + strlen(strnum) + strlen(s2) + 1); // +1 for the null-terminator

    // in real code you would check for errors in malloc here
    strcpy(result, s1);
    strcat(result, filename);
    strcat(result, strnum);
    strcat(result, s2);

    fp = fopen(result,"w");

    /* write lines of text into the file stream*/
    for(i = 0; i < length; i++){
        fprintf(fp, "%f\n",tobeplot[i]);
    }

    /* close the file*/
    fclose (fp);
    free(result);
}


void summary_array(char *print_name, float *arr, int n)
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
        if (arr[i] == 0){
           idxzero++;
        }
    }

    float mean=0;
    mean = sum / n;
    printf("%s || mean = %f; min=%f; max=%f; number of zeros=%f \n", print_name, mean, min, max, idxzero);
}


void make_network_CA(int n, float learning_rate, float momentum, float decay, int time_steps, int notruth, int batch, int subdivisions, int random, int adam, float B1, float B2, float eps, int h, int w, int c, int inputs, int max_crop, int min_crop, float max_ratio, float min_ratio, int center, float clip, float angle, float aspect, float saturation, float exposure, float hue, int burn_in, float power, int max_batches)
{
  TEEC_Operation op;
  uint32_t origin;
  TEEC_Result res;

    int passint[17];

    passint[0] = n;
    passint[1] = time_steps;
    passint[2] = notruth;
    passint[3] = batch;
    passint[4] = subdivisions;
    passint[5] = random;
    passint[6] = adam;
    passint[7] = h;
    passint[8] = w;
    passint[9] = c;
    passint[10] = inputs;
    passint[11] = max_crop;
    passint[12] = min_crop;
    passint[13] = center;
    passint[14] = burn_in;
    passint[15] = max_batches;

    float passfloat[15];
    passfloat[0] = learning_rate;
    passfloat[1] = momentum;
    passfloat[2] = decay;
    passfloat[3] = B1;
    passfloat[4] = B2;
    passfloat[5] = eps;
    passfloat[6] = max_ratio;
    passfloat[7] = min_ratio;
    passfloat[8] = clip;
    passfloat[9] = angle;
    passfloat[10] = aspect;
    passfloat[11] = saturation;
    passfloat[12] = exposure;
    passfloat[13] = hue;
    passfloat[14] = power;

    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT, TEEC_MEMREF_TEMP_INPUT,
                                     TEEC_NONE, TEEC_NONE);

    op.params[0].tmpref.buffer = passint;
    op.params[0].tmpref.size = sizeof(passint);

    op.params[1].tmpref.buffer = passfloat;
    op.params[1].tmpref.size = sizeof(passfloat);

    res = TEEC_InvokeCommand(&sess, MAKE_NETWORK_CMD,
                             &op, &origin);


    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_InvokeCommand(MAKE_NET) failed 0x%x origin 0x%x",
         res, origin);
}

void update_net_agrv_CA_allocateSM(int workspace_size, float *workspace)
{
    uint32_t origin;
    TEEC_Result res;
    workspaceSM.size  = sizeof(float) * workspace_size;
    workspaceSM.flags = TEEC_MEM_INPUT | TEEC_MEM_OUTPUT;

    res = TEEC_AllocateSharedMemory(
                     &ctx,
                     &workspaceSM);
     if (res != TEEC_SUCCESS)
     errx(1, "TEEC_InvokeCommand(UPDATE_NET_ASM) failed 0x%x origin 0x%x", res, origin);
}

void update_net_agrv_CA(int cond, int workspace_size, float *workspace)
{
    // forward condition
    if(cond == 0)
    {
        TEEC_Operation op;
        uint32_t origin;
        TEEC_Result res;

        workspaceSM.buffer = workspace;

        memset(&op, 0, sizeof(op));
        op.paramTypes = TEEC_PARAM_TYPES(TEEC_VALUE_INPUT, TEEC_MEMREF_PARTIAL_INOUT,
                                         TEEC_NONE, TEEC_NONE);

        op.params[0].value.a = cond;
        op.params[1].memref.parent = &workspaceSM;
        op.params[1].memref.offset = 0;
        op.params[1].memref.size   = sizeof(float) * workspace_size;

        res = TEEC_InvokeCommand(&sess, WORKSPACE_NETWORK_CMD,
                                 &op, &origin);

         if (res != TEEC_SUCCESS)
         errx(1, "TEEC_InvokeCommand(UPDATE_NET) failed 0x%x origin 0x%x",
              res, origin);
    }

    // backward condition
    if(cond == 1){
        float *wsbuffer = workspaceSM.buffer;
        for(int z=0; z<workspace_size; z++){
              workspace[z] = wsbuffer[z];
        }
    }
}

void make_shortcut_layer_CA(int batch, int index, int w, int h, int c, int from_out_w, int from_out_h, int from_out_c)
{
    TEEC_Operation op;
    uint32_t origin;
    TEEC_Result res;
    int passint[9];
    passint[0] = batch;
    passint[1] = index;
    passint[2] = w;
    passint[3] = h;
    passint[4] = c;
    passint[5] = from_out_w;
    passint[6] = from_out_h;
    passint[7] = from_out_c;
    passint[8] = current_layer_n;
    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT, TEEC_NONE,
                                     TEEC_NONE, TEEC_NONE);  
    op.params[0].tmpref.buffer = passint;
    op.params[0].tmpref.size = sizeof(passint);

    res = TEEC_InvokeCommand(&sess, MAKE_SHORTCUT_LAYER_CMD,
                             &op, &origin);
    if (res != TEEC_SUCCESS)
    {
        errx(1, "TEEC_InvokeCommand(CONV) failed 0x%x origin 0x%x",
            res, origin);

    }
}
void make_convolutional_layer_CA(int batch, int h, int w, int c, int n, int groups, int size, int stride, int padding, ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam, int flipped, float dot)
{
  TEEC_Operation op;
  uint32_t origin;
  TEEC_Result res;

    int passint[15];
    passint[0] = batch;
    passint[1] = h;
    passint[2] = w;
    passint[3] = c;
    passint[4] = n;
    passint[5] = groups;
    passint[6] = size;
    passint[7] = stride;
    passint[8] = padding;
    passint[9] = batch_normalize;
    passint[10] = binary;
    passint[11] = xnor;
    passint[12] = adam;
    passint[13] = flipped;
    passint[14] = current_layer_n;

    float passflo = dot;
    char *acti = get_activation_string(activation);

    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT, TEEC_VALUE_INPUT,
                                     TEEC_MEMREF_TEMP_INPUT, TEEC_NONE);

    op.params[0].tmpref.buffer = passint;
    op.params[0].tmpref.size = sizeof(passint);

    op.params[1].value.a = passflo;

    op.params[2].tmpref.buffer = acti;
    op.params[2].tmpref.size = strlen(acti)+1;

    res = TEEC_InvokeCommand(&sess, MAKE_CONV_CMD,
                             &op, &origin);


    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_InvokeCommand(CONV) failed 0x%x origin 0x%x",
         res, origin);
}

void make_maxpool_layer_CA(int batch, int h, int w, int c, int size, int stride, int padding)
{
  //invoke op and transfer paramters
  TEEC_Operation op;
  uint32_t origin;
  TEEC_Result res;

    int passint[8];
    passint[0] = batch;
    passint[1] = h;
    passint[2] = w;
    passint[3] = c;
    passint[4] = size;
    passint[5] = stride;
    passint[6] = padding;
    passint[7] = current_layer_n;

    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT, TEEC_NONE,
                                     TEEC_NONE, TEEC_NONE);

    op.params[0].tmpref.buffer = passint;
    op.params[0].tmpref.size = sizeof(passint);

    res = TEEC_InvokeCommand(&sess, MAKE_MAX_CMD,
                             &op, &origin);

    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_InvokeCommand(MAX) failed 0x%x origin 0x%x",
         res, origin);
}



void make_avgpool_layer_CA(int batch, int h, int w, int c)
{
  //invoke op and transfer paramters
  TEEC_Operation op;
  uint32_t origin;
  TEEC_Result res;

    int passint[5];
    passint[0] = batch;
    passint[1] = h;
    passint[2] = w;
    passint[3] = c;
    passint[4] = current_layer_n;
    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT, TEEC_NONE,
                                     TEEC_NONE, TEEC_NONE);

    op.params[0].tmpref.buffer = passint;
    op.params[0].tmpref.size = sizeof(passint);

    res = TEEC_InvokeCommand(&sess, MAKE_AVG_CMD,
                             &op, &origin);

    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_InvokeCommand(AVG) failed 0x%x origin 0x%x",
         res, origin);
}

void make_dropout_layer_CA(int batch, int inputs, float probability, int w, int h, int c, float *net_prev_output, float *net_prev_delta)
{
  //invoke op and transfer paramters
  TEEC_Operation op;
  uint32_t origin;
  TEEC_Result res;

    int passint[6];
    passint[0] = batch;
    passint[1] = inputs;
    passint[2] = w;
    passint[3] = h;
    passint[4] = c;
    passint[5] = current_layer_n;
    float passflo[1];
    passflo[0] = probability;

    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT, TEEC_MEMREF_TEMP_INPUT,
                                     TEEC_MEMREF_TEMP_INPUT, TEEC_MEMREF_TEMP_INPUT);

    op.params[0].tmpref.buffer = passint;
    op.params[0].tmpref.size = sizeof(passint);
    op.params[1].tmpref.buffer = passflo;
    op.params[1].tmpref.size = sizeof(float)*1;


    if(debug_plot_bool == 1){
        debug_plot("net_prev_output", sysCount, net_prev_output, inputs*batch);
        debug_plot("net_prev_delta", sysCount, net_prev_delta, inputs*batch);
    }


    op.params[2].tmpref.buffer = net_prev_output;
    op.params[2].tmpref.size = sizeof(float)*inputs*batch;
    op.params[3].tmpref.buffer = net_prev_delta;
    op.params[3].tmpref.size = sizeof(float)*inputs*batch;
////////////////////////

    res = TEEC_InvokeCommand(&sess, MAKE_DROP_CMD,
                             &op, &origin);

    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_InvokeCommand(DROP) failed 0x%x origin 0x%x",
         res, origin);
}



void make_connected_layer_CA(int batch, int inputs, int outputs, ACTIVATION activation, int batch_normalize, int adam)
{
    //invoke op and transfer paramters
    TEEC_Operation op;
    uint32_t origin;
    TEEC_Result res;

    int passarg[6];
    passarg[0] = batch;
    passarg[1] = inputs;
    passarg[2] = outputs;
    passarg[3] = batch_normalize;
    passarg[4] = adam;
    passarg[5] = current_layer_n;
    char *actv = get_activation_string(activation);

    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT, TEEC_MEMREF_TEMP_INPUT,
                                     TEEC_NONE, TEEC_NONE);

    op.params[0].tmpref.buffer = passarg;
    op.params[0].tmpref.size = sizeof(passarg);

    op.params[1].tmpref.buffer = actv;
    op.params[1].tmpref.size = strlen(actv)+1;

    res = TEEC_InvokeCommand(&sess, MAKE_CONNECTED_CMD,
                             &op, &origin);


    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_InvokeCommand(FC) failed 0x%x origin 0x%x",
         res, origin);
}

void make_softmax_layer_CA(int batch, int inputs, int groups, float temperature, int w, int h, int c, int spatial, int noloss)
{
    //invoke op and transfer paramters
    TEEC_Operation op;
    uint32_t origin;
    TEEC_Result res;

    int passint[9];
    float passflo = temperature;
    passint[0] = batch;
    passint[1] = inputs;
    passint[2] = groups;
    passint[3] = w;
    passint[4] = h;
    passint[5] = c;
    passint[6] = spatial;
    passint[7] = noloss;
    passint[8] = current_layer_n;
    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT,
        TEEC_VALUE_INPUT,
        TEEC_NONE, TEEC_NONE);

    op.params[0].tmpref.buffer = passint;
    op.params[0].tmpref.size = sizeof(passint);
    op.params[1].value.a = passflo;

    res = TEEC_InvokeCommand(&sess, MAKE_SOFTMAX_CMD,
                             &op, &origin);
    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_InvokeCommand(SOFTMAX) failed 0x%x origin 0x%x",
         res, origin);
}

void make_cost_layer_CA(int batch, int inputs, COST_TYPE cost_type, float scale, float ratio, float noobject_scale, float thresh)
{
    //invoke op and transfer paramters
    TEEC_Operation op;
    uint32_t origin;
    TEEC_Result res;

    int passint[3];
    float passflo[4];
    char *passcost;

    passint[0] = batch;
    passint[1] = inputs;
    passint[2] = current_layer_n;
    passflo[0] = scale;
    passflo[1] = ratio;
    passflo[2] = noobject_scale;
    passflo[3] = thresh;

    passcost = get_cost_string(cost_type);

    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT,
        TEEC_MEMREF_TEMP_INPUT,
        TEEC_MEMREF_TEMP_INPUT, TEEC_NONE);

    op.params[0].tmpref.buffer = passint;
    op.params[0].tmpref.size = sizeof(passint);

    op.params[1].tmpref.buffer = passflo;
    op.params[1].tmpref.size = sizeof(passflo);

    op.params[2].tmpref.buffer = passcost;
    op.params[2].tmpref.size = strlen(passcost)+1;

    res = TEEC_InvokeCommand(&sess, MAKE_COST_CMD,
                             &op, &origin);
    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_InvokeCommand(COST) failed 0x%x origin 0x%x",
         res, origin);
}

void transfer_weights_CA(float *vec, int length, int layer_i, char type, int additional)
{
    TEEC_Operation op;
    uint32_t origin;
    TEEC_Result res;

    int passint[3];
    passint[0] = length;
    passint[1] = layer_i;
    passint[2] = additional;

    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT, TEEC_MEMREF_TEMP_INPUT, TEEC_VALUE_INPUT, TEEC_NONE);

    op.params[0].tmpref.buffer = vec;
    op.params[0].tmpref.size = sizeof(float)*length;

    op.params[1].tmpref.buffer = passint;
    op.params[1].tmpref.size = sizeof(passint);

    op.params[2].value.a = type;

    res = TEEC_InvokeCommand(&sess, TRANS_WEI_CMD,
                             &op, &origin);
    if (res != TEEC_SUCCESS)
        errx(1, "TEEC_InvokeCommand(TRANS_WEI) failed 0x%x origin 0x%x",
             res, origin);
}

void save_weights_CA(float *vec, int length, int layer_i, char type)
{
    TEEC_Operation op;
    uint32_t origin;
    TEEC_Result res;

    int passint[2];
    passint[0] = length;
    passint[1] = layer_i;

    float *weights_back = malloc(sizeof(float) * length);

    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_OUTPUT, TEEC_MEMREF_TEMP_INPUT, TEEC_VALUE_INPUT, TEEC_NONE);

    op.params[0].tmpref.buffer = weights_back;
    op.params[0].tmpref.size = sizeof(float) * length;

    op.params[1].tmpref.buffer = passint;
    op.params[1].tmpref.size = sizeof(passint);

    op.params[2].value.a = type;

    res = TEEC_InvokeCommand(&sess, SAVE_WEI_CMD,
                             &op, &origin);

    for(int z=0; z<length; z++){
         vec[z] = weights_back[z];
    }

    free(weights_back);

    if (res != TEEC_SUCCESS)
        errx(1, "TEEC_InvokeCommand(SAVE_WEI) failed 0x%x origin 0x%x",
             res, origin);
}


void forward_network_CA_test(float *net_input, int size)
{
    //invoke op and transfer paramters
    TEEC_Operation op;
    uint32_t origin;
    TEEC_Result res;

    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT, TEEC_VALUE_INPUT,
                                     TEEC_NONE, TEEC_NONE);

    float *params0 = malloc(sizeof(float)*size);
    for(int z=0; z < size; z++){
        params0[z] = net_input[z];
    }
    int params1 = 0;

    op.params[0].tmpref.buffer = params0;
    op.params[0].tmpref.size = sizeof(float) * size;
    op.params[1].value.a = params1;

    /////////  debug_plot  /////////
    // if(debug_plot_bool == 1){
    //     debug_plot("forward_net_input_", sysCount, params0, l_inputs*net_batch);
    // }
    res = TEEC_InvokeCommand(&sess, FORWARD_CMD_TEST,
                             &op, &origin);

    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_InvokeCommand(forward) failed 0x%x origin 0x%x",
         res, origin);

    free(params0);
}


void forward_network_CA_fusion(float *net_input, int size, int channel)
{
    //invoke op and transfer paramters
    TEEC_Operation op;
    uint32_t origin;
    TEEC_Result res;

    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT, TEEC_VALUE_INPUT,
                                     TEEC_NONE, TEEC_NONE);

    float *params0 = malloc(sizeof(float)*channel);
    for(int z=0; z < size; z++){
        params0[z] = net_input[z];
    }

    op.params[0].tmpref.buffer = params0;
    op.params[0].tmpref.size = sizeof(float) * size;
    op.params[1].value.a = channel;
    op.params[1].value.b = size;

    /////////  debug_plot  /////////
    // if(debug_plot_bool == 1){
    //     debug_plot("forward_net_input_", sysCount, params0, l_inputs*net_batch);
    // }
    res = TEEC_InvokeCommand(&sess, FUSION_TEST,
                             &op, &origin);

    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_InvokeCommand(forward) failed 0x%x origin 0x%x",
         res, origin);

    free(params0);
}

void forward_network_CA(float *net_input, int l_inputs, int net_batch, int net_train)
{
    //invoke op and transfer paramters
    TEEC_Operation op;
    uint32_t origin;
    TEEC_Result res;

    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT, TEEC_VALUE_INPUT,
                                     TEEC_NONE, TEEC_NONE);

    float *params0 = malloc(sizeof(float)*l_inputs*net_batch);
    for(int z=0; z<l_inputs*net_batch; z++){
        params0[z] = net_input[z];
    }
    int params1 = net_train;

    op.params[0].tmpref.buffer = params0;
    op.params[0].tmpref.size = sizeof(float) * l_inputs*net_batch;
    op.params[1].value.a = params1;

    /////////  debug_plot  /////////
    if(debug_plot_bool == 1){
        debug_plot("forward_net_input_", sysCount, params0, l_inputs*net_batch);
    }
    res = TEEC_InvokeCommand(&sess, FORWARD_CMD,
                             &op, &origin);

    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_InvokeCommand(forward) failed 0x%x origin 0x%x",
         res, origin);

    free(params0);
}


// void forward_network_part_CA(float *net_input, int l_inputs, int net_batch, int net_train, int start_idx, int *next_idx)
// void forward_network_part_CA(float *net_input, int l_inputs, int net_batch, int net_train, int start_idx, int end_idx)
void forward_network_part_CA(float *net_input, int l_inputs, int net_batch, int net_train, int start_idx, int end_idx, float *output, int outputs)
{
    clock_t start = clock();
    //invoke op and transfer paramters
    TEEC_Operation op;
    uint32_t origin;
    TEEC_Result res;

    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT, TEEC_VALUE_INPUT,
                                     TEEC_VALUE_INPUT, TEEC_MEMREF_TEMP_OUTPUT);


    int params1 = net_train;
    op.params[0].tmpref.buffer = net_input;
    op.params[0].tmpref.size = sizeof(float) * l_inputs*net_batch;
    op.params[1].value.a = params1;
    op.params[2].value.a = start_idx;
    op.params[2].value.b = end_idx;
    op.params[3].tmpref.buffer = output;
    op.params[3].tmpref.size = sizeof(float) * outputs;

    // printf("[REE] l_inputs*net_batch: %d\n", l_inputs*net_batch);
    res = TEEC_InvokeCommand(&sess, FORWARD_CMD_PART,
                             &op, &origin);
    
    if (res != TEEC_SUCCESS)
        errx(1, "TEEC_InvokeCommand(forward part) failed 0x%x origin 0x%x", res, origin);
    clock_t end = clock();
    
    float * rt_value = op.params[3].tmpref.buffer;
    for (int i = 0; i < outputs; ++i)
    {
        output[i] = rt_value[i];
    }
    tee_total_us += end - start; 
}

void first_forward_network_part_CA(void){
    TEEC_Operation op;
    uint32_t origin;
    TEEC_Result res;

    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_NONE, TEEC_NONE,
                                     TEEC_NONE, TEEC_NONE);
    res = TEEC_InvokeCommand(&sess, FIRST_N_CMD,
                            &op, &origin);
    if (res != TEEC_SUCCESS)
        errx(1, "TEEC_InvokeCommand(first forward) failed 0x%x origin 0x%x", res, origin);               
}

void free_forward_network_part_CA(void){
    TEEC_Operation op;
    uint32_t origin;
    TEEC_Result res;

    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_NONE, TEEC_NONE,
                                     TEEC_NONE, TEEC_NONE);
    res = TEEC_InvokeCommand(&sess, FREE_N_CMD,
                            &op, &origin);
    if (res != TEEC_SUCCESS)
        errx(1, "TEEC_InvokeCommand(first forward) failed 0x%x origin 0x%x", res, origin);               
}
#define TEST_OBJECT_SIZE	(64 * 64 * 64)
// #define TEST_OBJECT_SIZE	(32 * 32 * 64)
// #define TEST_OBJECT_SIZE	(56 * 56 * 64)
// #define TEST_OBJECT_SIZE	(16 * 16 * 64)
// TODO:
void init_mask(void)
{
    //invoke op and transfer paramters
    TEEC_Operation op;
    uint32_t origin;
    TEEC_Result res;

    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT, TEEC_VALUE_INPUT,
                                     TEEC_NONE, TEEC_NONE);
    // op.paramTypes = TEEC_PARAM_TYPES(TEEC_NONE, TEEC_NONE,
    //                                  TEEC_NONE, TEEC_NONE);
    int size = TEST_OBJECT_SIZE;
    float *params0 = malloc(sizeof(float)*size);
    op.params[0].tmpref.buffer = params0;
    op.params[0].tmpref.size = sizeof(float) * size;
    op.params[1].value.a = size;

    /////////  debug_plot  /////////
    // if(debug_plot_bool == 1){
    //     debug_plot("forward_net_input_", sysCount, params0, l_inputs*net_batch);
    // }
    // printf("init_mask\n");
    res = TEEC_InvokeCommand(&sess, INIT_CMD_MASK,
                             &op, &origin);

    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_InvokeCommand(init mask) failed 0x%x origin 0x%x",
         res, origin);

    free(params0);
}

void free_mask(void)
{
    //invoke op and transfer paramters
    TEEC_Operation op;
    uint32_t origin;
    TEEC_Result res;

    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_NONE, TEEC_NONE,
                                     TEEC_NONE, TEEC_NONE);

    res = TEEC_InvokeCommand(&sess, FREE_CMD,
                             &op, &origin);

    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_InvokeCommand(free mask) failed 0x%x origin 0x%x",
         res, origin);
}
void get_tee_run_time(uint32_t *deobj_run_ms, uint32_t *mask_run_ms, uint32_t *tee_run_ms)
{
    TEEC_Operation op;
    uint32_t origin;
    TEEC_Result res;
    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_VALUE_OUTPUT, TEEC_VALUE_OUTPUT,
                                     TEEC_NONE, TEEC_NONE);
    res = TEEC_InvokeCommand(&sess, RETURN_RUN_TIME_CMD,
                            &op, &origin);
    if (res != TEEC_SUCCESS)
    {
        errx(1, "TEEC_InvokeCommand(get tee run time) failed 0x%x origin 0x%x",
            res, origin);
    }
    *deobj_run_ms = op.params[0].value.a;
    *mask_run_ms = op.params[0].value.b;
    *tee_run_ms = op.params[1].value.a;
}

void forward_network_per_layer_CA(layer_info_and_weights layer_info, float *input, int input_size, float *weights, float *biases, 
    float *scales, float *rolling_mean, float *variance){
    // layer_info.batch_normalize = 0;
    TEEC_Operation op;
    uint32_t origin;
    TEEC_Result res;
    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(
        TEEC_MEMREF_TEMP_INPUT, 
        TEEC_MEMREF_TEMP_INPUT,
        TEEC_MEMREF_TEMP_INPUT, 
        TEEC_MEMREF_TEMP_INPUT
    );


    float *params0;
    size_t params0_size = 0;
    float *params1;
    float *params2;
    layer_info_and_weights *params3;
    params3 = &layer_info; 
    size_t params1_size = sizeof(float) * layer_info.weights_length + sizeof(float) * layer_info.biases_length;
    size_t params2_size ;
    if (layer_info.batch_normalize)
    {
        if (layer_info.type == CONVOLUTIONAL || layer_info.type == CONNECTED)
        {
            params2_size = layer_info.biases_length * 3 * sizeof(float);
        }
        else
        {
            params2_size = 1;
        }
        params2 = (float *)malloc(params2_size);
    
        int params2_size_3 = layer_info.biases_length;
        for (int i = 0; i < params2_size_3; ++i)
        { 
            params2[i + params2_size_3 * 0] = scales[i];
        }

        for (int i = 0; i < params2_size_3; ++i)
        { 
            params2[i + params2_size_3 * 1] = rolling_mean[i];
        }
        for (int i = 0; i < params2_size_3; ++i)
        { 
            params2[i + params2_size_3 * 2] = variance[i];
        }

    }
    else{
        params2_size = 1;
        params2 = (float *)malloc(sizeof(float) * params2_size);
    }
    if (params1_size == 0)
    {
        params1_size = 1;
        params1 = (float *)malloc(sizeof(float) * params1_size);
    }
    else 
    {
        params1 = (float *)malloc(sizeof(float) * params1_size);
        for(int i = 0; i < layer_info.weights_length; ++i)
        {
            params1[i] = weights[i];
        }
        for(int i = 0; i < layer_info.biases_length; ++i)
        {
            params1[i + layer_info.weights_length] = biases[i];
        }
        
    }
    size_t params3_size = sizeof(layer_info_and_weights) * 1;
    // op.params[0].tmpref.buffer = params0;
    if (layer_info.need_input > 0) {
        params0 = input;
        params0_size = sizeof(float) * input_size;
        op.params[0].tmpref.buffer = params0;
        op.params[0].tmpref.size = params0_size;
    }
    else{
        params0 = (float *)malloc(1 *sizeof(float));
        params0_size = sizeof(float) * 1;
        op.params[0].tmpref.buffer = params0;
        op.params[0].tmpref.size = params0_size;
    }
    
    op.params[1].tmpref.buffer = params1;
    op.params[1].tmpref.size = params1_size;
    op.params[2].tmpref.buffer = params2;
    op.params[2].tmpref.size = params2_size;
    op.params[3].tmpref.buffer = params3;
    op.params[3].tmpref.size = params3_size;
// #define DEBUG_PER_LAYER
#ifdef DEBUG_PER_LAYER
    printf("[REE] type: %d, activation: %d, batch: %d, h: %d\n",
        (&layer_info)->type, (&layer_info)->activation, (&layer_info)->batch, (&layer_info)->h);
    printf("[REE] w: %d, c: %d, n: %d, groups: %d\n",
        (&layer_info)->w, (&layer_info)->c, (&layer_info)->n, (&layer_info)->groups);
    printf("[REE] size: %d, stride: %d, padding: %d, batch_normalize: %d\n",
        (&layer_info)->size, (&layer_info)->stride, (&layer_info)->padding, (&layer_info)->batch_normalize);
    printf("[REE] xnor: %d, binary: %d, adam: %d, flipped: %d\n",
        (&layer_info)->xnor, (&layer_info)->binary, (&layer_info)->adam, (&layer_info)->flipped);
    printf("[REE] cost_type: %d, biases_length: %d, weights_length: %d, inputs: %d\n",
        (&layer_info)->cost_type, (&layer_info)->biases_length, (&layer_info)->weights_length, (&layer_info)->inputs);
    printf("[REE] outputs: %d\n",
        (&layer_info)->outputs);   
    printf("[REE] input_size: %dB, nweights+nbiases: %dB, batch_parameters: %dB\n", op.params[0].tmpref.size, op.params[1].tmpref.size, op.params[2].tmpref.size);
    printf("[REE] layer:%d\n", layer_info.need_input);
#endif
    res = TEEC_InvokeCommand(&sess, FORWARD_CMD_PER_LAYER,
                            &op, &origin);
    if (res != TEEC_SUCCESS)
    {
        errx(1, "TEEC_InvokeCommand(forward network per layer) failed 0x%x origin 0x%x", res, origin);
    }
    if (layer_info.need_input <= 0){
        free(params0);
    }
    free(params1);
    free(params2);
    params2 = NULL;
    params1 = NULL;
    
}


void forward_connect_layer_part_CA(layer_info_and_weights layer_info, float *input, int input_size, float *weights, float *biases, 
    float *scales, float *rolling_mean, float *variance){
    // layer_info.batch_normalize = 0;
    TEEC_Operation op;
    uint32_t origin;
    TEEC_Result res;
    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(
        TEEC_MEMREF_TEMP_INPUT, 
        TEEC_MEMREF_TEMP_INPUT,
        TEEC_MEMREF_TEMP_INPUT, 
        TEEC_MEMREF_TEMP_INPUT
    );

// #define DEBUG_PER_LAYER
#ifdef DEBUG_PER_LAYER
    printf("[REE] type: %d, activation: %d, batch: %d, h: %d\n",
        (&layer_info)->type, (&layer_info)->activation, (&layer_info)->batch, (&layer_info)->h);
    printf("[REE] w: %d, c: %d, n: %d, groups: %d\n",
        (&layer_info)->w, (&layer_info)->c, (&layer_info)->n, (&layer_info)->groups);
    printf("[REE] size: %d, stride: %d, padding: %d, batch_normalize: %d\n",
        (&layer_info)->size, (&layer_info)->stride, (&layer_info)->padding, (&layer_info)->batch_normalize);
    printf("[REE] xnor: %d, binary: %d, adam: %d, flipped: %d\n",
        (&layer_info)->xnor, (&layer_info)->binary, (&layer_info)->adam, (&layer_info)->flipped);
    printf("[REE] cost_type: %d, biases_length: %d, weights_length: %d, inputs: %d\n",
        (&layer_info)->cost_type, (&layer_info)->biases_length, (&layer_info)->weights_length, (&layer_info)->inputs);
    printf("[REE] outputs: %d\n",
        (&layer_info)->outputs);   
    printf("[REE] layer_info: %dB, input_size: %dB, parameter: %dB\n", op.params[0].tmpref.size, op.params[2].tmpref.size, op.params[3].tmpref.size);
    printf("[REE] layer:%d\n", layer_info.need_input);
#endif
    float *params0;
    size_t params0_size = 0;
    float *params1;
    float *params2;
    layer_info_and_weights *params3;
    params3 = &layer_info; 
    size_t params1_size = sizeof(float) * layer_info.weights_length + sizeof(float) * layer_info.biases_length;
    size_t params2_size ;
    if (layer_info.batch_normalize)
    {
        if (layer_info.type == CONVOLUTIONAL || layer_info.type == CONNECTED)
        {
            params2_size = layer_info.biases_length * 3 * sizeof(float);
            // params2_size = layer_info.outputs * 3;
        }
        else
        {
            params2_size = 1;
        }
        // printf("[REE] params2_size: %d\n", params2_size);
        params2 = malloc(params2_size);
    
        // posix_memalign(&params2, 8, params2_size * sizeof(float));
        int params2_size_3 = layer_info.biases_length;
        for (int i = 0; i < params2_size_3; ++i)
        { 
            params2[i + params2_size_3 * 0] = scales[i];
        }

        for (int i = 0; i < params2_size_3; ++i)
        { 
            params2[i + params2_size_3 * 1] = rolling_mean[i];
        }
        for (int i = 0; i < params2_size_3; ++i)
        { 
            params2[i + params2_size_3 * 2] = variance[i];
        }

    }
    else{
        params2_size = 1;
        params2 = malloc(sizeof(float) * params2_size);
    }
    if (params1_size == 0)
    {
        params1_size = 1;
        params1 = malloc(sizeof(float) * params1_size);
    }
    else 
    {
        params1 = malloc(sizeof(float) * params1_size);
        for(int i = 0; i < layer_info.weights_length; ++i)
        {
            params1[i] = weights[i];
        }
        for(int i = 0; i < layer_info.biases_length; ++i)
        {
            params1[i + layer_info.weights_length] = biases[i];
        }
        
    }
    size_t params3_size = sizeof(layer_info_and_weights) * 1;
    // op.params[0].tmpref.buffer = params0;
    if (layer_info.need_input > 0) {
        params0 = input;
        params0_size = sizeof(float) * input_size;
        op.params[0].tmpref.buffer = params0;
        op.params[0].tmpref.size = params0_size;
    }
    else{
        params0 = malloc(1 *sizeof(float));
        params0_size = sizeof(float) * 1;
        op.params[0].tmpref.buffer = params0;
        op.params[0].tmpref.size = params0_size;
    }

    // if(layer_info.weights_length == 0){
    //     params1 = malloc(sizeof(float) * 1);
    //     params1_size = 1 * sizeof(float);
    // }
    // if(layer_info.biases_length == 0){
    //     params2 = malloc(sizeof(float) * 1);
    //     params2_size = 1 * sizeof(float);
    // }
    
    op.params[1].tmpref.buffer = params1;
    op.params[1].tmpref.size = params1_size;
    op.params[2].tmpref.buffer = params2;
    op.params[2].tmpref.size = params2_size;
    op.params[3].tmpref.buffer = params3;
    op.params[3].tmpref.size = params3_size;

    res = TEEC_InvokeCommand(&sess, FORWARD_CMD_CONNECT_LAYER_PART,
                            &op, &origin);
    free(params1);
    free(params2);
    params2 = NULL;
    params1 = NULL;
    if (res != TEEC_SUCCESS)
    {
        errx(1, "TEEC_InvokeCommand(forward network per layer) failed 0x%x origin 0x%x", res, origin);
    }
    if (layer_info.need_input <= 0){
        free(params0);
    }
    
}

void forward_network_CA_relu(float *output, int size, int activation, int out_channels, int size_y, int use_mask, int flops)
{
    clock_t start = clock();
    //invoke op and transfer paramters
    TEEC_Operation op;
    uint32_t origin;
    TEEC_Result res;
    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INOUT, TEEC_VALUE_INPUT,
                                     TEEC_VALUE_INOUT, TEEC_VALUE_INOUT);
    float *params0 = malloc(sizeof(float)*size);
    for(int z=0; z<size; z++){
        params0[z] = output[z];
    }

    op.params[0].tmpref.buffer = params0;
    op.params[0].tmpref.size = sizeof(float) * size;
    op.params[1].value.a = activation;
    op.params[1].value.b = out_channels;
    op.params[2].value.a = size_y;
    op.params[2].value.b = use_mask;
    op.params[3].value.a = flops;

    res = TEEC_InvokeCommand(&sess, FORWARD_CMD_RELU,
                             &op, &origin);

    if (res != TEEC_SUCCESS)
    {
        errx(1, "TEEC_InvokeCommand(forward relu) failed 0x%x origin 0x%x", res, origin);
    }
    
    for(int z=0; z<size; z++){
        output[z] = params0[z];
    }
    free(params0);
    clock_t end = clock();
    tee_total_us += end - start; 
    // printf("[GET_TIME] (flops, mask_t, deobj_t, relu_t):%d:%u:%u:%u:\n", flops, op.params[3].value.a, op.params[2].value.a, op.params[2].value.b);
}
void forward_network_CA_one_layer(float *net_input, int l_inputs, int net_batch, int net_train, int n_layer_th)
{
    //invoke op and transfer paramters
    TEEC_Operation op;
    uint32_t origin;
    TEEC_Result res;

    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT, TEEC_VALUE_INPUT,
                                     TEEC_NONE, TEEC_NONE);

    float *params0 = malloc(sizeof(float)*l_inputs*net_batch);
    for(int z=0; z<l_inputs*net_batch; z++){
        params0[z] = net_input[z];
    }
    int params1 = net_train;

    op.params[0].tmpref.buffer = params0;
    op.params[0].tmpref.size = sizeof(float) * l_inputs*net_batch;
    op.params[1].value.a = params1;
    op.params[1].value.b = n_layer_th;

    /////////  debug_plot  /////////
    if(debug_plot_bool == 1){
        debug_plot("forward_net_input_one_layer_", sysCount, params0, l_inputs*net_batch);
    }
    res = TEEC_InvokeCommand(&sess, FORWARD_CMD_NO_CONV_TEST,
                             &op, &origin);

    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_InvokeCommand(forward one layer) failed 0x%x origin 0x%x",
         res, origin);

    free(params0);
}


void tt_forward_network_back_one_layer_CA(float *l_output, int net_inputs, int net_batch, int n_layer_th)
{
  TEEC_Operation op;
  uint32_t origin;
  TEEC_Result res;

  net_input_back = malloc(sizeof(float) * net_inputs*net_batch);


  memset(&op, 0, sizeof(op));
  op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_OUTPUT, TEEC_NONE,
                                   TEEC_NONE, TEEC_NONE);



   op.params[0].tmpref.buffer = net_input_back;
   op.params[0].tmpref.size = sizeof(float) * net_inputs*net_batch;
   op.params[1].value.a = n_layer_th;

   res = TEEC_InvokeCommand(&sess, FORWARD_BACK_CMD_ONE_LAYER,
                            &op, &origin);

   for(int z=0; z<net_inputs * net_batch; z++){
       l_output[z] = net_input_back[z];
   }

   free(net_input_back);

   /////////  debug_plot  /////////
   if(debug_plot_bool == 1){
       debug_plot("forward_net_back_input_", sysCount, net_input_back, net_inputs*net_batch);
   }
   if (res != TEEC_SUCCESS)
   errx(1, "TEEC_InvokeCommand(forward_add_one_layer) failed 0x%x origin 0x%x",
        res, origin);
}


void make_activation_layer_CA(int batch, int inputs, ACTIVATION activation)
{
    //invoke op and transfer paramters
    TEEC_Operation op;
    uint32_t origin;
    TEEC_Result res;

    int passint[3];
    passint[0] = batch;
    passint[1] = inputs;
    passint[2] = current_layer_n;

    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT,
        TEEC_VALUE_INPUT,
        TEEC_NONE, TEEC_NONE);

    op.params[0].tmpref.buffer = passint;
    op.params[0].tmpref.size = sizeof(passint);
    char *acti = get_activation_string(activation);
    op.params[1].tmpref.buffer = acti;
    op.params[1].tmpref.size = strlen(acti)+1;

    res = TEEC_InvokeCommand(&sess, MAKE_ACTIVATE_CMD,
                             &op, &origin);
    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_InvokeCommand(ACTIVATION) failed 0x%x origin 0x%x",
         res, origin);
}



void make_batchnorm_layer_CA(int batch, int h, int w, int c)
{
    //invoke op and transfer paramters
    TEEC_Operation op;
    uint32_t origin;
    TEEC_Result res;

    int passint[5];
    passint[0] = batch;
    passint[1] = h;
    passint[2] = w;
    passint[3] = c;
    passint[4] = current_layer_n;

    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT,
        TEEC_NONE,
        TEEC_NONE, TEEC_NONE);

    op.params[0].tmpref.buffer = passint;
    op.params[0].tmpref.size = sizeof(passint);

    res = TEEC_InvokeCommand(&sess, MAKE_BATCHNORM_CMD,
                             &op, &origin);
    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_InvokeCommand(MAKE_BATCHNORM_CMD) failed 0x%x origin 0x%x",
         res, origin);
}


void forward_network_back_CA(float *l_output, int net_inputs, int net_batch)
{
  TEEC_Operation op;
  uint32_t origin;
  TEEC_Result res;
  float *net_input_back_temp;
  net_input_back_temp = malloc(sizeof(float) * net_inputs*net_batch);

  if (net_input_back_temp == NULL)
  {
    printf("[foward network back cA] malloc error\n");
  }
  memset(&op, 0, sizeof(op));
  op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_OUTPUT, TEEC_NONE,
                                   TEEC_NONE, TEEC_NONE);



   op.params[0].tmpref.buffer = net_input_back_temp;
   op.params[0].tmpref.size = sizeof(float) * net_inputs*net_batch;
    // printf("[REETOTEE] back start\n");

   res = TEEC_InvokeCommand(&sess, FORWARD_BACK_CMD,
                            &op, &origin);
   if (res != TEEC_SUCCESS)
   errx(1, "TEEC_InvokeCommand(forward_add) failed 0x%x origin 0x%x",
        res, origin);
    // printf("[REETOTEE] back(%d, %d)\n", net_inputs, net_batch);
   for(int z=0; z<net_inputs * net_batch; z++){
       l_output[z] = net_input_back_temp[z];
   }
    if (net_input_back_temp == NULL)
    {
        printf("[foward network back cA] no\n");
    }
    else{
       free(net_input_back_temp);
    }
    // printf("[REETOTEE] back2\n");
    //
}




void backward_network_CA(float *net_input, int l_inputs, int net_batch, int net_train)
{
    //invoke op and transfer paramters
    TEEC_Operation op;
    uint32_t origin;
    TEEC_Result res;


    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT, TEEC_VALUE_INPUT,
                                     TEEC_NONE, TEEC_NONE);

     float *params0 = malloc(sizeof(float)*l_inputs*net_batch);
     //float *params1 = malloc(sizeof(float)*l_inputs*net_batch);

     for(int z=0; z<l_inputs*net_batch; z++){
         params0[z] = net_input[z];
         //params1[z] = net_delta[z];
     }

    op.params[0].tmpref.buffer = params0; // as lta.output
    op.params[0].tmpref.size = sizeof(float)*l_inputs*net_batch;
    //op.params[1].tmpref.buffer = params1; // as n_delta
    //op.params[1].tmpref.size = sizeof(float)*l_inputs*net_batch;
    //op.params[2].value.a = net_train;
    op.params[1].value.a = net_train;

    res = TEEC_InvokeCommand(&sess, BACKWARD_CMD,
                             &op, &origin);

    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_InvokeCommand(backward) failed 0x%x origin 0x%x",
         res, origin);

     /////////  debug_plot  /////////
   if(debug_plot_bool == 1){
       debug_plot("backward_net_input_", sysCount, params0, l_inputs*net_batch);
       //debug_plot("backward_net_delta_", sysCount, params1, l_inputs*net_batch); // zero, removing!
   }
   free(params0);
   //free(params1);
}


void backward_network_CA_addidion(float *l_output, float *l_delta, int net_inputs, int net_batch)
{
  TEEC_Operation op;
  uint32_t origin;
  TEEC_Result res;

  net_input_back = malloc(sizeof(float) * net_inputs*net_batch);
  net_delta_back = malloc(sizeof(float) * net_inputs*net_batch);

  memset(&op, 0, sizeof(op));
  op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_OUTPUT, TEEC_MEMREF_TEMP_OUTPUT,
                                   TEEC_NONE, TEEC_NONE);

   op.params[0].tmpref.buffer = net_input_back;
   op.params[0].tmpref.size = sizeof(float) * net_inputs*net_batch;
   op.params[1].tmpref.buffer = net_delta_back;
   op.params[1].tmpref.size = sizeof(float) * net_inputs*net_batch;

   res = TEEC_InvokeCommand(&sess, BACKWARD_ADD_CMD,
                            &op, &origin);

   for(int z=0; z<net_inputs * net_batch; z++){
       l_output[z] = net_input_back[z];
       l_delta[z] = net_delta_back[z];
   }
   free(net_input_back);
   free(net_delta_back);


    /////////  debug_plot  /////////
   if(debug_plot_bool == 1){
       debug_plot("backward_net_add_input_", sysCount, net_input_back, net_inputs*net_batch);
       debug_plot("backward_net_add_delta_", sysCount, net_delta_back, net_inputs*net_batch);
   }
   if (res != TEEC_SUCCESS)
   errx(1, "TEEC_InvokeCommand(backward_add) failed 0x%x origin 0x%x",
        res, origin);
}



void backward_network_back_CA(float *net_input, int l_inputs, int net_batch, float *net_delta)
{
    //invoke op and transfer paramters
    TEEC_Operation op;
    uint32_t origin;
    TEEC_Result res;


    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT, TEEC_MEMREF_TEMP_INPUT,
                                     TEEC_NONE, TEEC_NONE);

    float *params0 = malloc(sizeof(float)*l_inputs*net_batch);
    float *params1 = malloc(sizeof(float)*l_inputs*net_batch);

    for(int z=0; z<l_inputs*net_batch; z++){
        params0[z] = net_input[z];
        params1[z] = net_delta[z];
    }

     /////////  debug_plot  /////////
    if(debug_plot_bool == 1){
        debug_plot("backward_net_back_input_", sysCount, params0, l_inputs*net_batch);
        debug_plot("backward_net_back_delta_", sysCount, params1, l_inputs*net_batch);
    }
    op.params[0].tmpref.buffer = params0; // as lta.output
    op.params[0].tmpref.size = sizeof(float)*l_inputs*net_batch;
    op.params[1].tmpref.buffer = params1; // as n_delta
    op.params[1].tmpref.size = sizeof(float)*l_inputs*net_batch;
#ifdef DEBUG
    printf("[REE] backward_network_back_TA_params:");
    for(int z=0; z<10; z++){
        printf("%f:%f ", params0[z], params1[z]);
    }
#endif

    res = TEEC_InvokeCommand(&sess, BACKWARD_BACK_CMD,
                             &op, &origin);

    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_InvokeCommand(backward_back) failed 0x%x origin 0x%x",
         res, origin);

   free(params0);
   free(params1);
}



void backward_network_back_CA_addidion(float *l_output, float *l_delta, int net_inputs, int net_batch)
{
  TEEC_Operation op;
  uint32_t origin;
  TEEC_Result res;

  net_input_back = malloc(sizeof(float) * net_inputs*net_batch);
  //net_delta_back = malloc(sizeof(float) * net_inputs*net_batch);


  memset(&op, 0, sizeof(op));
  op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_OUTPUT, TEEC_NONE,
                                   TEEC_NONE, TEEC_NONE);



   op.params[0].tmpref.buffer = net_input_back;
   op.params[0].tmpref.size = sizeof(float) * net_inputs*net_batch;
   //op.params[1].tmpref.buffer = net_delta_back;
   //op.params[1].tmpref.size = sizeof(float) * net_inputs*net_batch;

   res = TEEC_InvokeCommand(&sess, BACKWARD_BACK_ADD_CMD,
                            &op, &origin);

   for(int z=0; z<net_inputs * net_batch; z++){
       l_output[z] = net_input_back[z];
       //l_pp2.delta[z] = net_delta_back[z];
       l_delta[z] = 0.0f;
   }
   free(net_input_back);
   //free(net_delta_back);

   /////////  debug_plot  /////////
   if(debug_plot_bool == 1){
       debug_plot("backward_add_back_net_input_", sysCount, net_input_back, net_inputs*net_batch);
       //debug_plot("backward_add_back_net_delta_", sysCount, net_delta_back, net_inputs*net_batch); //zeros, removing!!
   }

   if (res != TEEC_SUCCESS)
   errx(1, "TEEC_InvokeCommand(backward_back_add) failed 0x%x origin 0x%x",
        res, origin);
}

void update_network_CA(update_args a)
{
    //invoke op and transfer paramters
    TEEC_Operation op;
    uint32_t origin;
    TEEC_Result res;

    int passint[3];
    passint[0] = a.batch;
    passint[1] = a.adam;
    passint[2] = a.t;

    float passflo[6];
    passflo[0] = a.learning_rate;
    passflo[1] = a.momentum;
    passflo[2] = a.decay;
    passflo[3] = a.B1;
    passflo[4] = a.B2;
    passflo[5] = a.eps;

    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT,
        TEEC_MEMREF_TEMP_INPUT,
        TEEC_NONE, TEEC_NONE);

    op.params[0].tmpref.buffer = passint;
    op.params[0].tmpref.size = sizeof(passint);
    op.params[1].tmpref.buffer = passflo;
    op.params[1].tmpref.size = sizeof(passflo);

    res = TEEC_InvokeCommand(&sess, UPDATE_CMD,
                             &op, &origin);
    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_InvokeCommand(update) failed 0x%x origin 0x%x",
         res, origin);
}



void net_truth_CA(float *net_truth, int net_truths, int net_batch)
{
    //invoke op and transfer paramters
    TEEC_Operation op;
    uint32_t origin;
    TEEC_Result res;

    // allocate memory for transmitting truth
    float *params0 = malloc(sizeof(float) * net_truths * net_batch);

    for(int z=0; z<net_truths*net_batch; z++){
        params0[z] = net_truth[z];
    }

    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT,
        TEEC_NONE,
        TEEC_NONE, TEEC_NONE);

    op.params[0].tmpref.buffer = params0;
    op.params[0].tmpref.size = sizeof(float)*net_truths*net_batch;

    res = TEEC_InvokeCommand(&sess, NET_TRUTH_CMD,
                             &op, &origin);
    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_InvokeCommand(truth) failed 0x%x origin 0x%x",
         res, origin);

     /////////  debug_plot  /////////
    if(debug_plot_bool == 1){
        debug_plot("backward_net_truth_", sysCount, net_truth, net_truths*net_batch);
    }
    free(params0);
}

void calc_network_loss_CA(int n, int batch)
{
    sysCount++;
    //invoke op and transfer paramters
    TEEC_Operation op;
    uint32_t origin;
    TEEC_Result res;

    int params0[2];
    params0[0] = n;
    params0[1] = batch;

    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT,
        TEEC_NONE,
        TEEC_NONE, TEEC_NONE);

    op.params[0].tmpref.buffer = params0;
    op.params[0].tmpref.size = sizeof(params0);

    res = TEEC_InvokeCommand(&sess, CALC_LOSS_CMD,
                             &op, &origin);
    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_InvokeCommand(loss) failed 0x%x origin 0x%x",
         res, origin);
}

void net_output_return_CA(int net_outputs, int net_batch)
{
    //invoke op and transfer paramters
    TEEC_Operation op;
    uint32_t origin;
    TEEC_Result res;

    net_output_back = malloc(sizeof(float) * net_outputs * net_batch);

    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_OUTPUT,
                                     TEEC_NONE,
                                     TEEC_NONE, TEEC_NONE);

    op.params[0].tmpref.buffer = net_output_back;
    op.params[0].tmpref.size = sizeof(float) * net_outputs * net_batch;

    res = TEEC_InvokeCommand(&sess, OUTPUT_RETURN_CMD,
                             &op, &origin);

    float *tem = op.params[0].tmpref.buffer;

    if (res != TEEC_SUCCESS)
        errx(1, "TEEC_InvokeCommand(return) failed 0x%x origin 0x%x",
             res, origin);
}


void prepare_tee_session()
{
    TEEC_UUID uuid = TA_DARKNETP_UUID;
    uint32_t origin;
    TEEC_Result res;

    /* Initialize a context connecting us to the TEE */
    res = TEEC_InitializeContext(NULL, &ctx);
    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_InitializeContext failed with code 0x%x", res);

    /* Open a session with the TA */
    res = TEEC_OpenSession(&ctx, &sess, &uuid,
                           TEEC_LOGIN_PUBLIC, NULL, NULL, &origin);
    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_Opensession failed with code 0x%x origin 0x%x",
         res, origin);
}

void terminate_tee_session()
{
    TEEC_CloseSession(&sess);
    TEEC_FinalizeContext(&ctx);
}



int main(int argc, char **argv)
{

    printf("Prepare session with the TA\n");
    prepare_tee_session();
    printf("Begin darknet\n");
    darknet_main(argc, argv);
    terminate_tee_session();
    return 0;
}
