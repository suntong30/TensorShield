export PYTHONPATH=../:$PYTHONPATH

for MODEL in resnet18 
    do
    for DATASET in CIFAR10
        do
        for MODE in  block_shallow
            do
            /home/gpu2/miniconda3/envs/sunt_torch2.1.1/bin/python3 \
            train_layer.py $DATASET $MODEL \
            -d 0 \
            -o results/train_layer/$DATASET-$MODEL \
            --transfer_path  transfer_data/$DATASET-$MODEL \
            --victim_dir victim_mia/$DATASET-$MODEL \
            --shadow_model_dir shadow_mia/$DATASET-$MODEL \
            --budgets 100,300 \
            --lr 0.1 \
            --remain-lr 1e-3 \
            --update-lr 1e-2 \
            --epochs 10 \
            --pretrained \
            --graybox-mode $MODE \
            --argmaxed 
        done
    done
done