#include "darknet.h"
#include "main.h"
#include "parser.h"

#include <sys/time.h>
#include <assert.h>
#include <sys/resource.h>
#include <unistd.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <string.h>

#include "tcp_transfer.h"


void getMemory(FILE *output_file) {

        // stores each word in status file
        char buffer[1024] = "";
        unsigned long vmsize, vmrss, vmdata, vmstk, vmexe, vmlib;
        FILE* file = fopen("/proc/self/status", "r");

        // read the entire file
        if(file) {
                while (fscanf(file, " %1023s", buffer) == 1) {
                        if (strcmp(buffer, "VmSize:") == 0) {
                                fscanf(file, " %ld", &vmsize);
                                printf("vmsize:%ld; ", vmsize);
                                fprintf(output_file, "vmsize:%ld; ", vmsize);
                        }
                        if (strcmp(buffer, "VmRSS:") == 0) {
                                fscanf(file, " %ld", &vmrss);
                                printf("vmrss:%ld; ", vmrss);
                                fprintf(output_file, "vmrss:%ld; ", vmrss);
                        }
                        if (strcmp(buffer, "VmData:") == 0) {
                                fscanf(file, " %ld", &vmdata);
                                printf("vmdata:%ld; ", vmdata);
                                fprintf(output_file, "vmdata:%ld; ", vmdata);
                        }
                        if (strcmp(buffer, "VmStk:") == 0) {
                                fscanf(file, " %ld", &vmstk);
                                printf("vmstk:%ld; ", vmstk);
                                fprintf(output_file, "vmstk:%ld; ", vmstk);
                        }
                        if (strcmp(buffer, "VmExe:") == 0) {
                                fscanf(file, " %ld", &vmexe);
                                printf("vmexe:%ld; ", vmexe);
                                fprintf(output_file, "vmexe:%ld; ", vmexe);
                        }
                        if (strcmp(buffer, "VmLib:") == 0) {
                                fscanf(file, " %ld", &vmlib);
                                printf("vmlib:%ld\n", vmlib);
                                fprintf(output_file, "vmlib:%ld\n", vmlib);
                        }
                }
        }else{
                printf("memory status file not found");
        }

        fclose(file);
}


float *get_regression_values(char **labels, int n)
{
        float *v = calloc(n, sizeof(float));
        int i;
        for(i = 0; i < n; ++i) {
                char *p = strchr(labels[i], ' ');
                *p = 0;
                v[i] = atof(p+1);
        }
        return v;
}


void train_classifier(char *datacfg, char *cfgfile, char *weightfile_o, int *gpus, int ngpus, int clear, bool fl)
{
        int i;

        float avg_loss = -1;
        char *base = basecfg(cfgfile);

        list *options = read_data_cfg(datacfg);
        char *backup_directory = option_find_str(options, "backup", "/backup/");
        char *weightfile = weightfile_o;

        printf("%s\n", base);
        printf("%d\n", ngpus);
        network **nets = calloc(ngpus, sizeof(network*));

        // save received net para
        if(fl) {
                printf("Federated Learning module: waiting for Server connection....\n");
                printf("\n");
                char buff[256];
                sprintf(buff, "%s/%s_fl.weights", backup_directory, base);

                // int rf_res = tcp_transfer(buff, "receive");
                tcp_transfer(buff, "receive");
                weightfile = buff;
        }

        srand(time(0));
        int seed = rand();

        for(i = 0; i < ngpus; ++i) {
                srand(seed);
#ifdef GPU
                if(gpu_index >= 0){
                opencl_set_device(i);
                }
#endif
                nets[i] = load_network(cfgfile, weightfile, clear);
                nets[i]->learning_rate *= ngpus;
        }
        srand(time(0));
        network *net = nets[0];

        int imgs = net->batch * net->subdivisions * ngpus;

        printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);

        int tag = option_find_int_quiet(options, "tag", 0);
        char *label_list = option_find_str(options, "labels", "data/labels.list");
        char *train_list = option_find_str(options, "train", "data/train.list");
        char *tree = option_find_str(options, "tree", 0);
        if (tree) net->hierarchy = read_tree(tree);
        int classes = option_find_int(options, "classes", 2);

        char **labels = 0;
        if(!tag) {
                labels = get_labels(label_list);
        }
        list *plist = get_paths(train_list);
        char **paths = (char **)list_to_array(plist);
        printf("%d\n", plist->size);
        int N = plist->size;
        double time;

        load_args args = {0};
        args.w = net->w;
        args.h = net->h;
        args.threads = 32;
        args.hierarchy = net->hierarchy;

        args.min = net->min_ratio*net->w;
        args.max = net->max_ratio*net->w;
        printf("%d %d\n", args.min, args.max);
        args.angle = net->angle;
        args.aspect = net->aspect;
        args.exposure = net->exposure;
        args.saturation = net->saturation;
        args.hue = net->hue;
        args.size = net->w;

        args.paths = paths;
        args.classes = classes;
        args.n = imgs;
        args.m = N;
        args.labels = labels;
        if (tag) {
                args.type = TAG_DATA;
        } else {
                args.type = CLASSIFICATION_DATA;
        }

        data train;
        data buffer;
        pthread_t load_thread;
        args.d = &buffer;
        load_thread = load_data(args);

        int count = 0;
        int epoch = (*net->seen)/N;

        // output file
        struct stat st = {0};
        if (stat("/media/results", &st) == -1) {
                mkdir("/media/results", 0700);
        }

        char delim[] = "/.";
        char *ptr = strtok(cfgfile, delim);
        ptr = strtok(NULL, delim);

        char pp_str_start[5];
        sprintf(pp_str_start, "%d", partition_point1 + 1);
        char pp_str_end[5];
        sprintf(pp_str_end, "%d", partition_point2);

        char *output_dir[80];
        strcpy(output_dir, "/media/results/train_");
        strcat(output_dir, ptr);
        strcat(output_dir, "_pps");
        strcat(output_dir, pp_str_start);
        strcat(output_dir, "_ppe");
        strcat(output_dir, pp_str_end);
        strcat(output_dir, ".txt");

        printf("output file: %s\n", output_dir);

        FILE *output_file = fopen(output_dir, "w");

        // additional train
        printf("current_batch=%d \n", get_current_batch(net));
        if(get_current_batch(net) >= net->max_batches) {
                net->max_batches = get_current_batch(net) + net->max_batches;
        }

        while(get_current_batch(net) < net->max_batches || net->max_batches == 0) {
                if(net->random && count++%40 == 0) {
                        printf("Resizing\n");
                        int dim = (rand() % 11 + 4) * 32;
                        //if (get_current_batch(net)+200 > net->max_batches) dim = 608;
                        //int dim = (rand() % 4 + 16) * 32;
                        printf("%d\n", dim);
                        args.w = dim;
                        args.h = dim;
                        args.size = dim;
                        args.min = net->min_ratio*dim;
                        args.max = net->max_ratio*dim;
                        printf("%d %d\n", args.min, args.max);

                        pthread_join(load_thread, 0);
                        train = buffer;
                        free_data(train);
                        load_thread = load_data(args);

                        for(i = 0; i < ngpus; ++i) {
                                resize_network(nets[i], dim, dim);
                        }
                        net = nets[0];
                }

                struct rusage usage;
                struct timeval startu, endu, starts, ends;

                getrusage(RUSAGE_SELF, &usage);
                startu = usage.ru_utime;
                starts = usage.ru_stime;

                time = what_time_is_it_now();

                pthread_join(load_thread, 0);
                train = buffer;
                load_thread = load_data(args);

                printf("Loaded: %lf seconds\n", what_time_is_it_now()-time);
                time = what_time_is_it_now();

                float loss = 0;
#ifdef GPU
                if (gpu_index >= 0) {
                        if (ngpus == 1) {
                                loss = train_network(net, train);
                        } else {
                                loss = train_networks(nets, ngpus, train, 4);
                        }
                }
                else {
                        loss = train_network(net, train);
                }
#else
                loss = train_network(net, train);
#endif
                if(avg_loss == -1) avg_loss = loss;
                avg_loss = avg_loss*.9 + loss*.1;
                printf("%ld, %.3f: %f, %f avg, %f rate, %lf seconds, %ld images\n", get_current_batch(net), (float)(*net->seen)/N, loss, avg_loss, get_current_rate(net), what_time_is_it_now()-time, *net->seen);
                fprintf(output_file, "%ld, %.3f: %f, %f avg, %f rate, %lf seconds, %ld images\n", get_current_batch(net), (float)(*net->seen)/N, loss, avg_loss, get_current_rate(net), what_time_is_it_now()-time, *net->seen);
// TODO

                getrusage(RUSAGE_SELF, &usage);
                endu = usage.ru_utime;
                ends = usage.ru_stime;
                printf("user CPU start: %lu.%06lu; end: %lu.%06lu\n", startu.tv_sec, startu.tv_usec, endu.tv_sec, endu.tv_usec);
                printf("kernel CPU start: %lu.%06lu; end: %lu.%06lu\n", starts.tv_sec, starts.tv_usec, ends.tv_sec, ends.tv_usec);
                printf("Max: %ld  kilobytes\n", usage.ru_maxrss);
                fprintf(output_file, "user CPU start: %lu.%06lu; end: %lu.%06lu\n", startu.tv_sec, startu.tv_usec, endu.tv_sec, endu.tv_usec);
                fprintf(output_file, "kernel CPU start: %lu.%06lu; end: %lu.%06lu\n", starts.tv_sec, starts.tv_usec, ends.tv_sec, ends.tv_usec);
                fprintf(output_file, "Max: %ld  kilobytes\n", usage.ru_maxrss);
                getMemory(output_file);


                free_data(train);
                // if(*net->seen/N > epoch) {
                //         epoch = *net->seen/N;
                //         char buff[256];
                //         sprintf(buff, "%s/%s_%d.weights",backup_directory,base, epoch);
                //         save_weights(net, buff);
                // }
                // if(get_current_batch(net)%1000 == 0) {
                //         char buff[256];
                //         sprintf(buff, "%s/%s.backup",backup_directory,base);
                //         save_weights(net, buff);
                // }
#ifdef GPU_STATS
                opencl_dump_mem_stat();
#endif
        }
        fclose(output_file);

        char buff[256];
        sprintf(buff, "%s/%s.weights", backup_directory, base);
        save_weights(net, buff);
        pthread_join(load_thread, 0);
        // save received net para
        // if(fl) {
        //         int sf_res = tcp_transfer(buff, "send");
        // }

        free_network(net);
        if(labels) free_ptrs((void**)labels, classes);
        free_ptrs((void**)paths, plist->size);
        free_list(plist);
        free(base);
}





void validate_classifier_crop(char *datacfg, char *filename, char *weightfile)
{
        int i = 0;
        network *net = load_network(filename, weightfile, 0);
        srand(time(0));

        list *options = read_data_cfg(datacfg);

        char *label_list = option_find_str(options, "labels", "data/labels.list");
        char *valid_list = option_find_str(options, "valid", "data/train.list");
        int classes = option_find_int(options, "classes", 2);
        int topk = option_find_int(options, "top", 1);

        char **labels = get_labels(label_list);
        list *plist = get_paths(valid_list);

        char **paths = (char **)list_to_array(plist);
        int m = plist->size;
        free_list(plist);

        clock_t time;
        float avg_acc = 0;
        float avg_topk = 0;
        int splits = m/1000;
        int num = (i+1)*m/splits - i*m/splits;

        data val, buffer;

        load_args args = {0};
        args.w = net->w;
        args.h = net->h;

        args.paths = paths;
        args.classes = classes;
        args.n = num;
        args.m = 0;
        args.labels = labels;
        args.d = &buffer;
        args.type = OLD_CLASSIFICATION_DATA;

        pthread_t load_thread = load_data_in_thread(args);
        for(i = 1; i <= splits; ++i) {
                time=clock();

                pthread_join(load_thread, 0);
                val = buffer;

                num = (i+1)*m/splits - i*m/splits;
                char **part = paths+(i*m/splits);
                if(i != splits) {
                        args.paths = part;
                        load_thread = load_data_in_thread(args);
                }
                printf("Loaded: %d images in %lf seconds\n", val.X.rows, sec(clock()-time));

                time=clock();
                float *acc = network_accuracies(net, val, topk);
                avg_acc += acc[0];
                avg_topk += acc[1];
                printf("%d: top 1: %f, top %d: %f, %lf seconds, %d images\n", i, avg_acc/i, topk, avg_topk/i, sec(clock()-time), val.X.rows);
                free_data(val);
        }
}

void validate_classifier_10(char *datacfg, char *filename, char *weightfile)
{
        int i, j;
        network *net = load_network(filename, weightfile, 0);
        set_batch_network(net, 1);
        srand(time(0));

        list *options = read_data_cfg(datacfg);

        char *label_list = option_find_str(options, "labels", "data/labels.list");
        char *valid_list = option_find_str(options, "valid", "data/train.list");
        int classes = option_find_int(options, "classes", 2);
        int topk = option_find_int(options, "top", 1);

        char **labels = get_labels(label_list);
        list *plist = get_paths(valid_list);

        char **paths = (char **)list_to_array(plist);
        int m = plist->size;
        free_list(plist);

        float avg_acc = 0;
        float avg_topk = 0;
        int *indexes = calloc(topk, sizeof(int));

        for(i = 0; i < m; ++i) {
                int class = -1;
                char *path = paths[i];
                for(j = 0; j < classes; ++j) {
                        if(strstr(path, labels[j])) {
                                class = j;
                                break;
                        }
                }
                int w = net->w;
                int h = net->h;
                int shift = 32;
                image im = load_image_color(paths[i], w+shift, h+shift);
                image images[10];
                images[0] = crop_image(im, -shift, -shift, w, h);
                images[1] = crop_image(im, shift, -shift, w, h);
                images[2] = crop_image(im, 0, 0, w, h);
                images[3] = crop_image(im, -shift, shift, w, h);
                images[4] = crop_image(im, shift, shift, w, h);
                flip_image(im);
                images[5] = crop_image(im, -shift, -shift, w, h);
                images[6] = crop_image(im, shift, -shift, w, h);
                images[7] = crop_image(im, 0, 0, w, h);
                images[8] = crop_image(im, -shift, shift, w, h);
                images[9] = crop_image(im, shift, shift, w, h);
                float *pred = calloc(classes, sizeof(float));
                for(j = 0; j < 10; ++j) {
                        float *p = network_predict(net, images[j].data);
                        if(net->hierarchy) hierarchy_predictions(p, net->outputs, net->hierarchy, 1, 1);
                        axpy_cpu(classes, 1, p, 1, pred, 1);
                        free_image(images[j]);
                }
                free_image(im);
                top_k(pred, classes, topk, indexes);
                free(pred);
                if(indexes[0] == class) avg_acc += 1;
                for(j = 0; j < topk; ++j) {
                        if(indexes[j] == class) avg_topk += 1;
                }

                printf("%d: top 1: %f, top %d: %f\n", i, avg_acc/(i+1), topk, avg_topk/(i+1));
        }
}

void validate_classifier_full(char *datacfg, char *filename, char *weightfile)
{
        int i, j;
        network *net = load_network(filename, weightfile, 0);
        set_batch_network(net, 1);
        srand(time(0));

        list *options = read_data_cfg(datacfg);

        char *label_list = option_find_str(options, "labels", "data/labels.list");
        char *valid_list = option_find_str(options, "valid", "data/train.list");
        int classes = option_find_int(options, "classes", 2);
        int topk = option_find_int(options, "top", 1);

        char **labels = get_labels(label_list);
        list *plist = get_paths(valid_list);

        char **paths = (char **)list_to_array(plist);
        int m = plist->size;
        free_list(plist);

        float avg_acc = 0;
        float avg_topk = 0;
        int *indexes = calloc(topk, sizeof(int));

        int size = net->w;
        for(i = 0; i < m; ++i) {
                int class = -1;
                char *path = paths[i];
                for(j = 0; j < classes; ++j) {
                        if(strstr(path, labels[j])) {
                                class = j;
                                break;
                        }
                }
                image im = load_image_color(paths[i], 0, 0);
                image resized = resize_min(im, size);
                resize_network(net, resized.w, resized.h);
                //show_image(im, "orig");
                //show_image(crop, "cropped");
                //cvWaitKey(0);
                float *pred = network_predict(net, resized.data);
                if(net->hierarchy) hierarchy_predictions(pred, net->outputs, net->hierarchy, 1, 1);

                free_image(im);
                free_image(resized);
                top_k(pred, classes, topk, indexes);

                if(indexes[0] == class) avg_acc += 1;
                for(j = 0; j < topk; ++j) {
                        if(indexes[j] == class) avg_topk += 1;
                }

                printf("%d: top 1: %f, top %d: %f\n", i, avg_acc/(i+1), topk, avg_topk/(i+1));
        }
}


void validate_classifier_single(char *datacfg, char *filename, char *weightfile)
{
        int i, j;
        network *net = load_network(filename, weightfile, 0);
        set_batch_network(net, 1);
        srand(time(0));

        list *options = read_data_cfg(datacfg);

        char *label_list = option_find_str(options, "labels", "data/labels.list");
        char *leaf_list = option_find_str(options, "leaves", 0);
        if(leaf_list) change_leaves(net->hierarchy, leaf_list);
        char *valid_list = option_find_str(options, "valid", "data/train.list");
        int classes = option_find_int(options, "classes", 2);
        int topk = option_find_int(options, "top", 1);

        char **labels = get_labels(label_list);
        list *plist = get_paths(valid_list);

        char **paths = (char **)list_to_array(plist);
        int m = plist->size;
        free_list(plist);

        float avg_acc = 0;
        float avg_topk = 0;
        int *indexes = calloc(topk, sizeof(int));

        for(i = 0; i < m; ++i) {
                int class = -1;
                char *path = paths[i];
                for(j = 0; j < classes; ++j) {
                        if(strstr(path, labels[j])) {
                                class = j;
                                break;
                        }
                }
                image im = load_image_color(paths[i], 0, 0);
                image crop = center_crop_image(im, net->w, net->h);
                //grayscale_image_3c(crop);
                //show_image(im, "orig");
                //show_image(crop, "cropped");
                //cvWaitKey(0);
                float *pred = network_predict(net, crop.data);
                if(net->hierarchy) hierarchy_predictions(pred, net->outputs, net->hierarchy, 1, 1);

                free_image(im);
                free_image(crop);
                top_k(pred, classes, topk, indexes);

                if(indexes[0] == class) avg_acc += 1;
                for(j = 0; j < topk; ++j) {
                        if(indexes[j] == class) avg_topk += 1;
                }

                printf("%s, %d, %f, %f, \n", paths[i], class, pred[0], pred[1]);
                printf("%d: top 1: %f, top %d: %f\n", i, avg_acc/(i+1), topk, avg_topk/(i+1));
        }
}


void validate_classifier_multi(char *datacfg, char *cfg, char *weights)
{
        int i, j;
        network *net = load_network(cfg, weights, 0);
        set_batch_network(net, 1);
        srand(time(0));

        list *options = read_data_cfg(datacfg);

        char *label_list = option_find_str(options, "labels", "data/labels.list");
        char *valid_list = option_find_str(options, "valid", "data/train.list");
        int classes = option_find_int(options, "classes", 2);
        int topk = option_find_int(options, "top", 1);

        char **labels = get_labels(label_list);
        list *plist = get_paths(valid_list);
        //int scales[] = {224, 288, 320, 352, 384};
        int scales[] = {224, 256, 288, 320};
        int nscales = sizeof(scales)/sizeof(scales[0]);

        char **paths = (char **)list_to_array(plist);
        int m = plist->size;
        free_list(plist);

        float avg_acc = 0;
        float avg_topk = 0;
        int *indexes = calloc(topk, sizeof(int));

        for(i = 0; i < m; ++i) {
                int class = -1;
                char *path = paths[i];
                for(j = 0; j < classes; ++j) {
                        if(strstr(path, labels[j])) {
                                class = j;
                                break;
                        }
                }
                float *pred = calloc(classes, sizeof(float));
                image im = load_image_color(paths[i], 0, 0);
                for(j = 0; j < nscales; ++j) {
                        image r = resize_max(im, scales[j]);
                        resize_network(net, r.w, r.h);
                        float *p = network_predict(net, r.data);
                        if(net->hierarchy) hierarchy_predictions(p, net->outputs, net->hierarchy, 1, 1);
                        axpy_cpu(classes, 1, p, 1, pred, 1);
                        flip_image(r);
                        p = network_predict(net, r.data);
                        axpy_cpu(classes, 1, p, 1, pred, 1);
                        if(r.data != im.data) free_image(r);
                }
                free_image(im);
                top_k(pred, classes, topk, indexes);
                free(pred);
                if(indexes[0] == class) avg_acc += 1;
                for(j = 0; j < topk; ++j) {
                        if(indexes[j] == class) avg_topk += 1;
                }

                printf("%d: top 1: %f, top %d: %f\n", i, avg_acc/(i+1), topk, avg_topk/(i+1));
        }
}

void try_classifier(char *datacfg, char *cfgfile, char *weightfile, char *filename, int layer_num)
{
        network *net = load_network(cfgfile, weightfile, 0);
        set_batch_network(net, 1);
        srand(2222222);

        list *options = read_data_cfg(datacfg);

        char *name_list = option_find_str(options, "names", 0);
        if(!name_list) name_list = option_find_str(options, "labels", "data/labels.list");
        int top = option_find_int(options, "top", 1);

        int i = 0;
        char **names = get_labels(name_list);
        clock_t time;
        int *indexes = calloc(top, sizeof(int));
        char buff[256];
        char *input = buff;
        while(1) {
                if(filename) {
                        strncpy(input, filename, 256);
                }else{
                        printf("Enter Image Path: ");
                        fflush(stdout);
                        input = fgets(input, 256, stdin);
                        if(!input) return;
                        strtok(input, "\n");
                }
                image orig = load_image_color(input, 0, 0);
                image r = resize_min(orig, 256);
                image im = crop_image(r, (r.w - 224 - 1)/2 + 1, (r.h - 224 - 1)/2 + 1, 224, 224);
                float mean[] = {0.48263312050943, 0.45230225481413, 0.40099074308742};
                float std[] = {0.22590347483426, 0.22120921437787, 0.22103996251583};
                float var[3];
                var[0] = std[0]*std[0];
                var[1] = std[1]*std[1];
                var[2] = std[2]*std[2];

                normalize_cpu(im.data, mean, var, 1, 3, im.w*im.h);

                float *X = im.data;
                time=clock();
                float *predictions = network_predict(net, X);

                layer l = net->layers[layer_num];
                for(i = 0; i < l.c; ++i) {
                        if(l.rolling_mean) printf("%f %f %f\n", l.rolling_mean[i], l.rolling_variance[i], l.scales[i]);
                }
#ifdef GPU
                if(gpu_index >= 0) {
                        opencl_pull_array(l.output_gpu, l.output, l.outputs);
                }
#endif
                for(i = 0; i < l.outputs; ++i) {
                        printf("%f\n", l.output[i]);
                }
                /*

                   printf("\n\nWeights\n");
                   for(i = 0; i < l.n*l.size*l.size*l.c; ++i){
                   printf("%f\n", l.filters[i]);
                   }

                   printf("\n\nBiases\n");
                   for(i = 0; i < l.n; ++i){
   cdd/f..f.sd.fasfda               printf("%f\n", l.biases[i]);
                   }
                 */

                top_predictions(net, top, indexes);
                printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
                for(i = 0; i < top; ++i) {
                        int index = indexes[i];
                        printf("%s: %f\n", names[index], predictions[index]);
                }
                free_image(im);
                if (filename) break;
        }
}

void print_run_ms(){
        uint32_t tee_operator_ms;
        uint32_t mask_run_ms;
        uint32_t deobf_run_ms;
        get_tee_run_time(&deobf_run_ms, &mask_run_ms, &tee_operator_ms);
        float mask_run_s = (float) mask_run_ms;
        mask_run_s /= 1000;
        float tee_operator_s = (float) tee_operator_ms;
        tee_operator_s /= 1000;
        float deobf_run_s = (float) deobf_run_ms;
        deobf_run_s /= 1000;
        float total_run_s = sec(total_run_us);
        float total_tee_run_s = sec(tee_total_us);
        float comm_run_s = total_tee_run_s - tee_operator_s;
        float cal_s = total_run_s - comm_run_s - mask_run_s - deobf_run_s;
        printf("===================================\n");
        printf("[RESULT] MASK: %f(s),\tDEOBJ: %f(s),\t COMM: %f(s)\t, CAL: %f(s).\n",
                mask_run_s, deobf_run_s, comm_run_s, cal_s);
        printf("[RESULT] MASK: %u(ms),\tDEOBJ: %u(ms),\t TEE OPERAOTR: %u(ms)\t, TOTLA: %f(s), TEE: %f(s).\n",
                mask_run_ms, deobf_run_ms, tee_operator_ms, total_run_s, total_tee_run_s);
        
        printf("====================================\n");
}

void predict_classifier(char *datacfg, char *cfgfile, char *weightfile, char *filename, int top)
{
        clock_t load_start_time = clock();
        network *net = load_network(cfgfile, weightfile, 0);
        clock_t load_end_time = clock();
        printf("[PREDICT CLASSIFIER] MODEL LOADER: %lf seconds\n", sec(load_end_time - load_start_time));
        set_batch_network(net, 1);
        srand(2222222);

        list *options = read_data_cfg(datacfg);

        char *name_list = option_find_str(options, "names", 0);
        if(!name_list) name_list = option_find_str(options, "labels", "data/labels.list");
        if(top == 0) top = option_find_int(options, "top", 1);

        int i = 0;
        char **names = get_labels(name_list);
        clock_t time;
        int *indexes = calloc(top, sizeof(int));
        char buff[256];
        char *input = buff;
        printf("[TOTAL PARAMS]: %d\n", total_params);
        if (use_tee_relu > 0){
                clock_t mask_e;
                clock_t mask_s = clock();
                init_mask();
                mask_e = clock();
                printf("[REE] INIT MASK: %lf (s)\n", sec(mask_e - mask_s));
        }

        while(1) {
                if(filename) {
                        strncpy(input, filename, 256);
                }else{
                        printf("Enter Image Path: ");
                        fflush(stdout);
                        input = fgets(input, 256, stdin);
                        if(!input) return;
                        strtok(input, "\n");
                }
                image im = load_image_color(input, 0, 0);
                image r = letterbox_image(im, net->w, net->h);
                float *X = r.data;
                // warmup
                // if (net->gpu_index >= 0)
                // {
                //         float *warmup_x = (float *)calloc(net->inputs, sizeof(float));
                //         for(int warmup_idx = 0; warmup_idx < 2; ++warmup_idx)
                //         {
                //                 network_predict(net, warmup_x);
                //         }
                //         free(warmup_x);
                // }

                time=clock();
                
                float *predictions;
                for (int _run_idx = 0; _run_idx < run_predict_time; ++_run_idx)
                {
                        predictions = network_predict(net, X);
                }
                total_run_us = clock() - time;
                if(use_tee_relu > 0)
                {
                        print_run_ms();
                }
                if(net->hierarchy) hierarchy_predictions(predictions, net->outputs, net->hierarchy, 1, 1);
                printf("%s: AAA Predicted in %f seconds.\n", input, sec(clock()-time));
                if (use_tee_relu > 0){
                        clock_t mask_e;
                        clock_t mask_s = clock();
                        free_mask();
                        mask_e = clock();
                        printf("[REE] FREE MASK: %lf (s)\n", sec(mask_e - mask_s));
                }

                top_k(predictions, net->outputs, top, indexes);
                if (net_output_back != NULL)
                {
                        free(net_output_back);
                }
                return;
                struct rusage usage;
                struct timeval startu, endu, starts, ends;

                getrusage(RUSAGE_SELF, &usage);
                startu = usage.ru_utime;
                starts = usage.ru_stime;

                // output file
                struct stat st = {0};
                if (stat("/media/results", &st) == -1) {
                        mkdir("/media/results", 0700);
                }

                char delim[] = "/.";
                char *ptr = strtok(cfgfile, delim);
                ptr = strtok(NULL, delim);

                char pp_str_start[5];
                sprintf(pp_str_start, "%d", partition_point1 + 1);
                char pp_str_end[5];
                sprintf(pp_str_end, "%d", partition_point2);

                char *output_dir[80];
                strcpy(output_dir, "/media/results/predict_");
                strcat(output_dir, ptr);
                strcat(output_dir, "_pps");
                strcat(output_dir, pp_str_start);
                strcat(output_dir, "_ppe");
                strcat(output_dir, pp_str_end);
                strcat(output_dir, ".txt");

                printf("output file: %s\n", output_dir);
                FILE *output_file = fopen(output_dir, "a");

                fprintf(stderr, "%s: ADD Write File Predicted in %f seconds.\n", input, sec(clock()-time));
                fprintf(output_file, "%s: Predicted in %f seconds.\n", input, sec(clock()-time));

                for(i = 0; i < top; ++i) {
                        int index = indexes[i];
                        //if(net->hierarchy) printf("%d, %s: %f, parent: %s \n",index, names[index], predictions[index], (net->hierarchy->parent[index] >= 0) ? names[net->hierarchy->parent[index]] : "Root");
                        //else printf("%s: %f\n",names[index], predictions[index]);
                        printf("%5.2f%%: %s\n", predictions[index]*100, names[index]);
                }



                getrusage(RUSAGE_SELF, &usage);
                endu = usage.ru_utime;
                ends = usage.ru_stime;
                printf("user CPU start: %lu.%06lu; end: %lu.%06lu\n", startu.tv_sec, startu.tv_usec, endu.tv_sec, endu.tv_usec);
                printf("kernel CPU start: %lu.%06lu; end: %lu.%06lu\n", starts.tv_sec, starts.tv_usec, ends.tv_sec, ends.tv_usec);
                printf("Max: %ld  kilobytes\n", usage.ru_maxrss);
                fprintf(output_file, "user CPU start: %lu.%06lu; end: %lu.%06lu\n", startu.tv_sec, startu.tv_usec, endu.tv_sec, endu.tv_usec);
                fprintf(output_file, "kernel CPU start: %lu.%06lu; end: %lu.%06lu\n", starts.tv_sec, starts.tv_usec, ends.tv_sec, ends.tv_usec);
                fprintf(output_file, "Max: %ld  kilobytes\n", usage.ru_maxrss);
                getMemory(output_file);

                fclose(output_file);

                if(r.data != im.data) free_image(r);
                free_image(im);
                if (filename) break;
        }
        // if (use_tee_relu > 0){
        //         clock_t mask_e;
        //         clock_t mask_s = clock();
        //         free_mask();
        //         mask_e = clock();
        //         printf("[REE] FREE MASK: %lf (s)\n", sec(mask_e - mask_s));
        // }
}

void predict_classifier_demo(char *datacfg, char *cfgfile, char *weightfile, char *filename, int top)
{
        clock_t load_start_time = clock();
        network *net = load_network(cfgfile, weightfile, 0);
        clock_t load_end_time = clock();
        printf("[PREDICT CLASSIFIER] MODEL LOADER: %lf seconds\n", sec(load_end_time - load_start_time));
        set_batch_network(net, 1);
        srand(2222222);

        list *options = read_data_cfg(datacfg);

        char *name_list = option_find_str(options, "names", 0);
        if(!name_list) name_list = option_find_str(options, "labels", "data/labels.list");
        if(top == 0) top = option_find_int(options, "top", 1);

        int i = 0;
        char **names = get_labels(name_list);
        clock_t time;
        int *indexes = calloc(top, sizeof(int));
        char buff[256];
        char *input = buff;
        if (use_tee_relu > 0){
                clock_t mask_e;
                clock_t mask_s = clock();
                init_mask();
                mask_e = clock();
                printf("[REE] INIT MASK: %lf (s)\n", sec(mask_e - mask_s));
        }

        while(1) {
                if(filename) {
                        strncpy(input, filename, 256);
                }else{
                        printf("Enter Image Path: ");
                        fflush(stdout);
                        input = fgets(input, 256, stdin);
                        if(!input) return;
                        strtok(input, "\n");
                }
                image im = load_image_color(input, 0, 0);
                image r = letterbox_image(im, net->w, net->h);
                float *X = r.data;
                first_forward_network_part_CA();
                printf("[FORWARD]\n");
                // warmup
                // if (net->gpu_index >= 0)
                // {
                //         float *warmup_x = (float *)calloc(net->inputs, sizeof(float));
                //         for(int warmup_idx = 0; warmup_idx < 2; ++warmup_idx)
                //         {
                //                 network_predict(net, warmup_x);
                //         }
                //         free(warmup_x);
                // }
                time=clock();
                float *predictions;
                for (int _run_idx = 0; _run_idx < run_predict_time; ++_run_idx)
                {
                        predictions = network_predict_demo(net, X);
                }

                // float *predictions = network_predict_demo(net, X);
                // if(net->hierarchy) hierarchy_predictions(predictions, net->outputs, net->hierarchy, 1, 1);
                total_run_us = clock() - time;
                if(use_tee_relu > 0 || run_demo_idx > 0)
                {
                print_run_ms();
                }
                printf("%s: AAA Predicted in %f seconds.\n", input, sec(clock()-time));
                uint32_t tee_operator_ms;
                uint32_t mask_run_ms;
                uint32_t deobf_run_ms;
                get_tee_run_time(&deobf_run_ms, &mask_run_ms, &tee_operator_ms);
                
                free_forward_network_part_CA();
                printf("[REE] net->outputs: %d %d\n", net->outputs, net->layers[net->n - 1].outputs);
                // for(int i = 0; i < net->outputs; ++i){
                //         if(i % 100 == 0){
                //                 printf("\n");
                //         }
                //         printf("%f ", predictions[i]);
                // }
                // printf("\n");
                top_k(predictions, net->outputs, top, indexes);

                free(net_output_back);

                struct rusage usage;
                struct timeval startu, endu, starts, ends;

                getrusage(RUSAGE_SELF, &usage);
                startu = usage.ru_utime;
                starts = usage.ru_stime;

                // output file
                struct stat st = {0};
                if (stat("/media/results", &st) == -1) {
                        mkdir("/media/results", 0700);
                }

                char delim[] = "/.";
                char *ptr = strtok(cfgfile, delim);
                ptr = strtok(NULL, delim);

                char pp_str_start[5];
                sprintf(pp_str_start, "%d", partition_point1 + 1);
                char pp_str_end[5];
                sprintf(pp_str_end, "%d", partition_point2);

                char *output_dir[80];
                strcpy(output_dir, "/media/results/predict_");
                strcat(output_dir, ptr);
                strcat(output_dir, "_pps");
                strcat(output_dir, pp_str_start);
                strcat(output_dir, "_ppe");
                strcat(output_dir, pp_str_end);
                strcat(output_dir, ".txt");

                printf("output file: %s\n", output_dir);
                FILE *output_file = fopen(output_dir, "a");

                fprintf(stderr, "%s: ADD Write File Predicted in %f seconds.\n", input, sec(clock()-time));
                fprintf(output_file, "%s: Predicted in %f seconds.\n", input, sec(clock()-time));

                for(i = 0; i < top; ++i) {
                        int index = indexes[i];
                        //if(net->hierarchy) printf("%d, %s: %f, parent: %s \n",index, names[index], predictions[index], (net->hierarchy->parent[index] >= 0) ? names[net->hierarchy->parent[index]] : "Root");
                        //else printf("%s: %f\n",names[index], predictions[index]);
                        printf("%5.2f%%: %s\n", predictions[index]*100, names[index]);
                }



                getrusage(RUSAGE_SELF, &usage);
                endu = usage.ru_utime;
                ends = usage.ru_stime;
                printf("user CPU start: %lu.%06lu; end: %lu.%06lu\n", startu.tv_sec, startu.tv_usec, endu.tv_sec, endu.tv_usec);
                printf("kernel CPU start: %lu.%06lu; end: %lu.%06lu\n", starts.tv_sec, starts.tv_usec, ends.tv_sec, ends.tv_usec);
                printf("Max: %ld  kilobytes\n", usage.ru_maxrss);
                fprintf(output_file, "user CPU start: %lu.%06lu; end: %lu.%06lu\n", startu.tv_sec, startu.tv_usec, endu.tv_sec, endu.tv_usec);
                fprintf(output_file, "kernel CPU start: %lu.%06lu; end: %lu.%06lu\n", starts.tv_sec, starts.tv_usec, ends.tv_sec, ends.tv_usec);
                fprintf(output_file, "Max: %ld  kilobytes\n", usage.ru_maxrss);
                getMemory(output_file);

                fclose(output_file);

                if(r.data != im.data) free_image(r);
                free_image(im);
                if (filename) break;
        }
        if (use_tee_relu > 0){
                clock_t mask_e;
                clock_t mask_s = clock();
                free_mask();
                mask_e = clock();
                printf("[REE] FREE MASK: %lf (s)\n", sec(mask_e - mask_s));


        }
}



void predict_classifier_per_layer(char *datacfg, char *cfgfile, char *weightfile, char *filename, int top)
{
        clock_t load_start_time = clock();
        network *net = load_network_per_layer(cfgfile, weightfile, 0);
        clock_t load_end_time = clock();
        printf("[PREDICT CLASSIFIER] MODEL LOADER: %lf seconds\n", sec(load_end_time - load_start_time));
        set_batch_network(net, 1);
        srand(2222222);
        int i = 0;
        clock_t time;
        char buff[256];
        char *input = buff;
        while(1) {
                if(filename) {
                        strncpy(input, filename, 256);
                }else{
                        printf("Enter Image Path: ");
                        fflush(stdout);
                        input = fgets(input, 256, stdin);
                        if(!input) return;
                        strtok(input, "\n");
                }
                image im = load_image_color(input, 0, 0);
                image r = letterbox_image(im, net->w, net->h);
                float *X = r.data;
                // warmup
                // if (net->gpu_index >= 0)
                // {
                //         float *warmup_x = (float *)calloc(net->inputs, sizeof(float));
                //         for(int warmup_idx = 0; warmup_idx < 2; ++warmup_idx)
                //         {
                //                 network_predict(net, warmup_x);
                //         }
                //         free(warmup_x);
                // }

                time=clock();
                float *predictions;
                for (int _run_idx = 0; _run_idx < run_predict_time; ++_run_idx)
                {
                        predictions = tt_network_predict(net, X);
                }
                // float *predictions = tt_network_predict(net, X);
                // if(net->hierarchy) hierarchy_predictions(predictions, net->outputs, net->hierarchy, 1, 1);
                total_run_us = clock() - time;
                if(use_tee_relu > 0)
                {
                        print_run_ms();
                }
                printf("%s: AAA Predicted in %f seconds.\n", input, sec(clock()-time));
                free(net_output_back);

                struct rusage usage;
                struct timeval startu, endu, starts, ends;

                getrusage(RUSAGE_SELF, &usage);
                startu = usage.ru_utime;
                starts = usage.ru_stime;

                // output file
                struct stat st = {0};
                if (stat("/media/results", &st) == -1) {
                        mkdir("/media/results", 0700);
                }

                char delim[] = "/.";
                char *ptr = strtok(cfgfile, delim);
                ptr = strtok(NULL, delim);

                char pp_str_start[5];
                sprintf(pp_str_start, "%d", partition_point1 + 1);
                char pp_str_end[5];
                sprintf(pp_str_end, "%d", partition_point2);

                char *output_dir[80];
                strcpy(output_dir, "/media/results/predict_");
                strcat(output_dir, ptr);
                strcat(output_dir, "_pps");
                strcat(output_dir, pp_str_start);
                strcat(output_dir, "_ppe");
                strcat(output_dir, pp_str_end);
                strcat(output_dir, ".txt");

                printf("output file: %s\n", output_dir);
                FILE *output_file = fopen(output_dir, "a");

                fprintf(stderr, "%s: ADD Write File Predicted in %f seconds.\n", input, sec(clock()-time));
                fprintf(output_file, "%s: Predicted in %f seconds.\n", input, sec(clock()-time));
                getrusage(RUSAGE_SELF, &usage);
                endu = usage.ru_utime;
                ends = usage.ru_stime;
                printf("user CPU start: %lu.%06lu; end: %lu.%06lu\n", startu.tv_sec, startu.tv_usec, endu.tv_sec, endu.tv_usec);
                printf("kernel CPU start: %lu.%06lu; end: %lu.%06lu\n", starts.tv_sec, starts.tv_usec, ends.tv_sec, ends.tv_usec);
                printf("Max: %ld  kilobytes\n", usage.ru_maxrss);
                fprintf(output_file, "user CPU start: %lu.%06lu; end: %lu.%06lu\n", startu.tv_sec, startu.tv_usec, endu.tv_sec, endu.tv_usec);
                fprintf(output_file, "kernel CPU start: %lu.%06lu; end: %lu.%06lu\n", starts.tv_sec, starts.tv_usec, ends.tv_sec, ends.tv_usec);
                fprintf(output_file, "Max: %ld  kilobytes\n", usage.ru_maxrss);
                getMemory(output_file);

                fclose(output_file);

                if(r.data != im.data) free_image(r);
                free_image(im);
                if (filename) break;
        }
}



void label_classifier(char *datacfg, char *filename, char *weightfile)
{
        int i;
        network *net = load_network(filename, weightfile, 0);
        set_batch_network(net, 1);
        srand(time(0));

        list *options = read_data_cfg(datacfg);

        char *label_list = option_find_str(options, "names", "data/labels.list");
        char *test_list = option_find_str(options, "test", "data/train.list");
        int classes = option_find_int(options, "classes", 2);

        char **labels = get_labels(label_list);
        list *plist = get_paths(test_list);

        char **paths = (char **)list_to_array(plist);
        int m = plist->size;
        free_list(plist);

        for(i = 0; i < m; ++i) {
                image im = load_image_color(paths[i], 0, 0);
                image resized = resize_min(im, net->w);
                image crop = crop_image(resized, (resized.w - net->w)/2, (resized.h - net->h)/2, net->w, net->h);
                float *pred = network_predict(net, crop.data);
                if(resized.data != im.data) free_image(resized);
                free_image(im);
                free_image(crop);
                int ind = max_index(pred, classes);
                printf("%s\n", labels[ind]);
        }
}

void csv_classifier(char *datacfg, char *cfgfile, char *weightfile)
{
        int i,j;
        network *net = load_network(cfgfile, weightfile, 0);
        srand(time(0));

        list *options = read_data_cfg(datacfg);

        char *test_list = option_find_str(options, "test", "data/test.list");
        int top = option_find_int(options, "top", 1);

        list *plist = get_paths(test_list);

        char **paths = (char **)list_to_array(plist);
        int m = plist->size;
        free_list(plist);
        int *indexes = calloc(top, sizeof(int));

        for(i = 0; i < m; ++i) {
                double time = what_time_is_it_now();
                char *path = paths[i];
                image im = load_image_color(path, 0, 0);
                image r = letterbox_image(im, net->w, net->h);
                float *predictions = network_predict(net, r.data);
                if(net->hierarchy) hierarchy_predictions(predictions, net->outputs, net->hierarchy, 1, 1);
                top_k(predictions, net->outputs, top, indexes);

                printf("%s", path);
                for(j = 0; j < top; ++j) {
                        printf("\t%d", indexes[j]);
                }
                printf("\n");

                free_image(im);
                free_image(r);

                fprintf(stderr, "%lf seconds, %d images, %d total\n", what_time_is_it_now() - time, i+1, m);
        }
}

void test_classifier(char *datacfg, char *cfgfile, char *weightfile, int target_layer)
{
        int curr = 0;
        network *net = load_network(cfgfile, weightfile, 0);
        srand(time(0));

        list *options = read_data_cfg(datacfg);

        char *test_list = option_find_str(options, "test", "data/test.list");
        int classes = option_find_int(options, "classes", 2);

        list *plist = get_paths(test_list);

        char **paths = (char **)list_to_array(plist);
        int m = plist->size;
        free_list(plist);

        clock_t time;

        data val, buffer;

        load_args args = {0};
        args.w = net->w;
        args.h = net->h;
        args.paths = paths;
        args.classes = classes;
        args.n = net->batch;
        args.m = 0;
        args.labels = 0;
        args.d = &buffer;
        args.type = OLD_CLASSIFICATION_DATA;

        pthread_t load_thread = load_data_in_thread(args);
        for(curr = net->batch; curr < m; curr += net->batch) {
                time=clock();

                pthread_join(load_thread, 0);
                val = buffer;

                if(curr < m) {
                        args.paths = paths + curr;
                        if (curr + net->batch > m) args.n = m - curr;
                        load_thread = load_data_in_thread(args);
                }
                fprintf(stderr, "Loaded: %d images in %lf seconds\n", val.X.rows, sec(clock()-time));

                time=clock();
                matrix pred = network_predict_data(net, val);

                int i, j;
                if (target_layer >= 0) {
                        //layer l = net->layers[target_layer];
                }

                for(i = 0; i < pred.rows; ++i) {
                        printf("%s", paths[curr-net->batch+i]);
                        for(j = 0; j < pred.cols; ++j) {
                                printf("\t%g", pred.vals[i][j]);
                        }
                        printf("\n");
                }

                free_matrix(pred);

                fprintf(stderr, "%lf seconds, %d images, %d total\n", sec(clock()-time), val.X.rows, curr);
                free_data(val);
        }
}

void file_output_classifier(char *datacfg, char *filename, char *weightfile, char *listfile)
{
        int i,j;
        network *net = load_network(filename, weightfile, 0);
        set_batch_network(net, 1);
        srand(time(0));

        list *options = read_data_cfg(datacfg);

        //char *label_list = option_find_str(options, "names", "data/labels.list");
        int classes = option_find_int(options, "classes", 2);

        list *plist = get_paths(listfile);

        char **paths = (char **)list_to_array(plist);
        int m = plist->size;
        free_list(plist);

        for(i = 0; i < m; ++i) {
                image im = load_image_color(paths[i], 0, 0);
                image resized = resize_min(im, net->w);
                image crop = crop_image(resized, (resized.w - net->w)/2, (resized.h - net->h)/2, net->w, net->h);

                float *pred = network_predict(net, crop.data);
                if(net->hierarchy) hierarchy_predictions(pred, net->outputs, net->hierarchy, 0, 1);

                if(resized.data != im.data) free_image(resized);
                free_image(im);
                free_image(crop);

                printf("%s", paths[i]);
                for(j = 0; j < classes; ++j) {
                        printf("\t%g", pred[j]);
                }
                printf("\n");
        }
}


void threat_classifier(char *datacfg, char *cfgfile, char *weightfile, int cam_index, const char *filename)
{
#ifdef OPENCV
        float threat = 0;
        float roll = .2;

        printf("Classifier Demo\n");
        network *net = load_network(cfgfile, weightfile, 0);
        set_batch_network(net, 1);
        list *options = read_data_cfg(datacfg);

        srand(2222222);
        void * cap = open_video_stream(filename, cam_index, 0,0,0);

        int top = option_find_int(options, "top", 1);

        char *name_list = option_find_str(options, "names", 0);
        char **names = get_labels(name_list);

        int *indexes = calloc(top, sizeof(int));

        if(!cap) error("Couldn't connect to webcam.\n");
        //cvNamedWindow("Threat", CV_WINDOW_NORMAL);
        //cvResizeWindow("Threat", 512, 512);
        float fps = 0;
        int i;

        int count = 0;

        while(1) {
                ++count;
                struct timeval tval_before, tval_after, tval_result;
                gettimeofday(&tval_before, NULL);

                image in = get_image_from_stream(cap);
                if(!in.data) break;
                image in_s = resize_image(in, net->w, net->h);

                image out = in;
                int x1 = out.w / 20;
                int y1 = out.h / 20;
                int x2 = 2*x1;
                int y2 = out.h - out.h/20;

                int border = .01*out.h;
                int h = y2 - y1 - 2*border;
                int w = x2 - x1 - 2*border;

                float *predictions = network_predict(net, in_s.data);
                float curr_threat = 0;
                if(1) {
                        curr_threat = predictions[0] * 0 +
                                      predictions[1] * .6 +
                                      predictions[2];
                } else {
                        curr_threat = predictions[218] +
                                      predictions[539] +
                                      predictions[540] +
                                      predictions[368] +
                                      predictions[369] +
                                      predictions[370];
                }
                threat = roll * curr_threat + (1-roll) * threat;

                draw_box_width(out, x2 + border, y1 + .02*h, x2 + .5 * w, y1 + .02*h + border, border, 0,0,0);
                if(threat > .97) {
                        draw_box_width(out,  x2 + .5 * w + border,
                                       y1 + .02*h - 2*border,
                                       x2 + .5 * w + 6*border,
                                       y1 + .02*h + 3*border, 3*border, 1,0,0);
                }
                draw_box_width(out,  x2 + .5 * w + border,
                               y1 + .02*h - 2*border,
                               x2 + .5 * w + 6*border,
                               y1 + .02*h + 3*border, .5*border, 0,0,0);
                draw_box_width(out, x2 + border, y1 + .42*h, x2 + .5 * w, y1 + .42*h + border, border, 0,0,0);
                if(threat > .57) {
                        draw_box_width(out,  x2 + .5 * w + border,
                                       y1 + .42*h - 2*border,
                                       x2 + .5 * w + 6*border,
                                       y1 + .42*h + 3*border, 3*border, 1,1,0);
                }
                draw_box_width(out,  x2 + .5 * w + border,
                               y1 + .42*h - 2*border,
                               x2 + .5 * w + 6*border,
                               y1 + .42*h + 3*border, .5*border, 0,0,0);

                draw_box_width(out, x1, y1, x2, y2, border, 0,0,0);
                for(i = 0; i < threat * h; ++i) {
                        float ratio = (float) i / h;
                        float r = (ratio < .5) ? (2*(ratio)) : 1;
                        float g = (ratio < .5) ? 1 : 1 - 2*(ratio - .5);
                        draw_box_width(out, x1 + border, y2 - border - i, x2 - border, y2 - border - i, 1, r, g, 0);
                }
                top_predictions(net, top, indexes);
                char buff[256];
                sprintf(buff, "/home/pjreddie/tmp/threat_%06d", count);
                //save_image(out, buff);

                printf("\033[2J");
                printf("\033[1;1H");
                printf("\nFPS:%.0f\n",fps);

                for(i = 0; i < top; ++i) {
                        int index = indexes[i];
                        printf("%.1f%%: %s\n", predictions[index]*100, names[index]);
                }

                if(1) {
                        show_image(out, "Threat", 10);
                }
                free_image(in_s);
                free_image(in);

                gettimeofday(&tval_after, NULL);
                timersub(&tval_after, &tval_before, &tval_result);
                float curr = 1000000.f/((long int)tval_result.tv_usec);
                fps = .9*fps + .1*curr;
        }
#endif
}


void gun_classifier(char *datacfg, char *cfgfile, char *weightfile, int cam_index, const char *filename)
{
#ifdef OPENCV
        int bad_cats[] = {218, 539, 540, 1213, 1501, 1742, 1911, 2415, 4348, 19223, 368, 369, 370, 1133, 1200, 1306, 2122, 2301, 2537, 2823, 3179, 3596, 3639, 4489, 5107, 5140, 5289, 6240, 6631, 6762, 7048, 7171, 7969, 7984, 7989, 8824, 8927, 9915, 10270, 10448, 13401, 15205, 18358, 18894, 18895, 19249, 19697};

        printf("Classifier Demo\n");
        network *net = load_network(cfgfile, weightfile, 0);
        set_batch_network(net, 1);
        list *options = read_data_cfg(datacfg);

        srand(2222222);
        void * cap = open_video_stream(filename, cam_index, 0,0,0);

        int top = option_find_int(options, "top", 1);

        char *name_list = option_find_str(options, "names", 0);
        char **names = get_labels(name_list);

        int *indexes = calloc(top, sizeof(int));

        if(!cap) error("Couldn't connect to webcam.\n");
        float fps = 0;
        int i;

        while(1) {
                struct timeval tval_before, tval_after, tval_result;
                gettimeofday(&tval_before, NULL);

                image in = get_image_from_stream(cap);
                image in_s = resize_image(in, net->w, net->h);

                float *predictions = network_predict(net, in_s.data);
                top_predictions(net, top, indexes);

                printf("\033[2J");
                printf("\033[1;1H");

                int threat = 0;
                for(i = 0; i < sizeof(bad_cats)/sizeof(bad_cats[0]); ++i) {
                        int index = bad_cats[i];
                        if(predictions[index] > .01) {
                                printf("Threat Detected!\n");
                                threat = 1;
                                break;
                        }
                }
                if(!threat) printf("Scanning...\n");
                for(i = 0; i < sizeof(bad_cats)/sizeof(bad_cats[0]); ++i) {
                        int index = bad_cats[i];
                        if(predictions[index] > .01) {
                                printf("%s\n", names[index]);
                        }
                }

                show_image(in, "Threat Detection", 10);
                free_image(in_s);
                free_image(in);

                gettimeofday(&tval_after, NULL);
                timersub(&tval_after, &tval_before, &tval_result);
                float curr = 1000000.f/((long int)tval_result.tv_usec);
                fps = .9*fps + .1*curr;
        }
#endif
}

void demo_classifier(char *datacfg, char *cfgfile, char *weightfile, int cam_index, const char *filename)
{
#ifdef OPENCV
        char *base = basecfg(cfgfile);
        image **alphabet = load_alphabet();
        printf("Classifier Demo\n");
        network *net = load_network(cfgfile, weightfile, 0);
        set_batch_network(net, 1);
        list *options = read_data_cfg(datacfg);

        srand(2222222);

        int w = 1280;
        int h = 720;
        void * cap = open_video_stream(filename, cam_index, w, h, 0);

        int top = option_find_int(options, "top", 1);

        char *label_list = option_find_str(options, "labels", 0);
        char *name_list = option_find_str(options, "names", label_list);
        char **names = get_labels(name_list);

        int *indexes = calloc(top, sizeof(int));

        if(!cap) error("Couldn't connect to webcam.\n");
        float fps = 0;
        int i;

        while(1) {
                struct timeval tval_before, tval_after, tval_result;
                gettimeofday(&tval_before, NULL);

                image in = get_image_from_stream(cap);
                //image in_s = resize_image(in, net->w, net->h);
                image in_s = letterbox_image(in, net->w, net->h);

                float *predictions = network_predict(net, in_s.data);
                if(net->hierarchy) hierarchy_predictions(predictions, net->outputs, net->hierarchy, 1, 1);
                top_predictions(net, top, indexes);

                printf("\033[2J");
                printf("\033[1;1H");
                printf("\nFPS:%.0f\n",fps);

                int lh = in.h*.03;
                int toph = 3*lh;

                float rgb[3] = {1,1,1};
                for(i = 0; i < top; ++i) {
                        printf("%d\n", toph);
                        int index = indexes[i];
                        printf("%.1f%%: %s\n", predictions[index]*100, names[index]);

                        char buff[1024];
                        sprintf(buff, "%3.1f%%: %s\n", predictions[index]*100, names[index]);
                        image label = get_label(alphabet, buff, lh);
                        draw_label(in, toph, lh, label, rgb);
                        toph += 2*lh;
                        free_image(label);
                }

                show_image(in, base, 10);
                free_image(in_s);
                free_image(in);

                gettimeofday(&tval_after, NULL);
                timersub(&tval_after, &tval_before, &tval_result);
                float curr = 1000000.f/((long int)tval_result.tv_usec);
                fps = .9*fps + .1*curr;
        }
#endif
}



void run_classifier(int argc, char **argv)
{
        if(argc < 4) {
                fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
                return;
        }

        char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
        int ngpus;
        int *gpus = read_intlist(gpu_list, &ngpus, gpu_index);

        // partition point of DNN
        int pp_start = find_int_arg(argc, argv, "-pp_start", 999);
        if(pp_start == 999){ // when using pp_start_f for forzen first layers outside TEE
            pp_start = find_int_arg(argc, argv, "-pp_start_f", 999);
            frozen_bool = 1;
        }
        if(pp_start == 999){ // when using pp_f_only for forzen first layers (all in REE)
            pp_start = find_int_arg(argc, argv, "-pp_f_only", 999);
            frozen_bool = 2;
        }

        partition_point1 = pp_start - 1;
        int pp_end = find_int_arg(argc, argv, "-pp_end", 999);
        partition_point2 = pp_end;
        int dp = find_int_arg(argc, argv, "-dp", -1);
        global_dp = dp;

        sepa_save_bool = find_int_arg(argc, argv, "-ss", 0);
        // 0 no separate save and load
        // 1 separate save and load
        // 2 separate save but load together

        int cam_index = find_int_arg(argc, argv, "-c", 0);
        int top = find_int_arg(argc, argv, "-t", 0);
        int clear = find_arg(argc, argv, "-clear");
        char *data = argv[3];
        char *cfg = argv[4];
        char *weights = (argc > 5) ? argv[5] : 0;
        char *filename = (argc > 6) ? argv[6] : 0;
        char *layer_s = (argc > 7) ? argv[7] : 0;
        int layer = layer_s ? atoi(layer_s) : -1;
        if(0==strcmp(argv[2], "predict")) {
                state = 'p';
                predict_classifier(data, cfg, weights, filename, top);
        }
        else if(0==strcmp(argv[2], "predict_per_layer")) {
                state = 'p';
                predict_classifier_per_layer(data, cfg, weights, filename, top);
        }
        else if(0==strcmp(argv[2], "predict_demo")) {
                state = 'p';
                predict_classifier_demo(data, cfg, weights, filename, top);
        }
        else if(0==strcmp(argv[2], "fout")) file_output_classifier(data, cfg, weights, filename);
        else if(0==strcmp(argv[2], "try")) try_classifier(data, cfg, weights, filename, atoi(layer_s));
        else if(0==strcmp(argv[2], "train")) train_classifier(data, cfg, weights, gpus, ngpus, clear, false);
        else if(0==strcmp(argv[2], "train_fl")) train_classifier(data, cfg, weights, gpus, ngpus, clear, true);
        else if(0==strcmp(argv[2], "demo")) demo_classifier(data, cfg, weights, cam_index, filename);
        else if(0==strcmp(argv[2], "gun")) gun_classifier(data, cfg, weights, cam_index, filename);
        else if(0==strcmp(argv[2], "threat")) threat_classifier(data, cfg, weights, cam_index, filename);
        else if(0==strcmp(argv[2], "test")) test_classifier(data, cfg, weights, layer);
        else if(0==strcmp(argv[2], "csv")) csv_classifier(data, cfg, weights);
        else if(0==strcmp(argv[2], "label")) label_classifier(data, cfg, weights);
        else if(0==strcmp(argv[2], "valid")) validate_classifier_single(data, cfg, weights);
        else if(0==strcmp(argv[2], "validmulti")) validate_classifier_multi(data, cfg, weights);
        else if(0==strcmp(argv[2], "valid10")) validate_classifier_10(data, cfg, weights);
        else if(0==strcmp(argv[2], "validcrop")) validate_classifier_crop(data, cfg, weights);
        else if(0==strcmp(argv[2], "validfull")) validate_classifier_full(data, cfg, weights);
}
