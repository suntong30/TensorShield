#ifndef NETWORK_TA_H
#define NETWORK_TA_H
struct layer_info_and_weights {
    int index;
    int w2;
    int c2;
    int h2;
    int need_input;
    int type;
    int activation;
    int batch;
    int h;
    int w;
    int c;
    int n;
    int groups;
    int size;
    int stride;
    int padding;
    int batch_normalize;
    int xnor;
    int binary;
    int adam;
    float dot;
    int flipped;
    // COST LAYER
    int cost_type;
    float scale;
    float ratio;
    float noobject_scale;
    float thresh;
    // softmax layer
    float temperature;
    int spatial;
    int noloss;
    // DROPOUT
    float probability;
    int netnum;
    // TODO:
    int biases_length;
    int weights_length;
    int scales_length;
    int rolling_mean_length;
    int rolling_variance_length;
    int inputs;
    int outputs;
    int last_layer;
    int workspace_size;
    int finish;
};
// #define PRINT_TIME_DEBUG
// #define PRINT_TIME
typedef struct layer_info_and_weights layer_info_and_weights;
extern network_TA netta;
extern float *ta_net_input;
extern float *ta_net_delta;
extern float *ta_net_output;
extern float *only_one_input;
extern uint32_t mask_time_ms;
extern uint32_t deobf_time_ms;
extern uint32_t tee_operator_ms;
extern uint32_t mask_time_ms_flops;
extern uint32_t deobf_time_ms_flops;
extern uint32_t relu_time_ms_flops;
void make_network_TA(int n, float learning_rate, float momentum, float decay, int time_steps, int notruth, int batch, int subdivisions, int random, int adam, float B1, float B2, float eps, int h, int w, int c, int inputs, int max_crop, int min_crop, float max_ratio, float min_ratio, int center, float clip, float angle, float aspect, float saturation, float exposure, float hue, int burn_in, float power, int max_batches);
void calc_network_cost_TA();

void calc_network_loss_TA(int n, int batch);
void tt_forward_network_TA(int n_layer_th);
void forward_network_TA();
void forward_network_TA_TEST();
void forward_network_TA_DEDefusion(int out_channels, int size_y);
// void tt_forward_relu(float *output, int size, int activation);
// void tt_forward_relu(float *output, int size, int activation, int out_channels, int size_y, int use_mask);
void tt_forward_relu(float *output, int size, int activation, int out_channels, int size_y, int use_mask, int flops);
// void forward_network_per_layer(float *input, int need_input, layer_info_and_weights *layer_info);
// void forward_network_per_layer(float *input, float *weights, float *biases, layer_info_and_weights *layer_info);
void forward_network_per_layer(float *input, float *base_parameters, float *batch_parameters, layer_info_and_weights *layer_info);
void run_connected_layer_part(layer_info_and_weights *layer_info, float *base_parameters, float *batch_parameters);
void backward_network_TA(float *ca_net_input);
void write_mask_test(float *obj1_data, int size);
void free_simulate_arry();
void update_network_TA(update_args_TA a);
void other_read_raw_object(float *load_data, int size);
#endif
