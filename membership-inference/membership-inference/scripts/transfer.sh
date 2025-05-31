export PYTHONPATH=../:$PYTHONPATH

for MODEL in resnet18 
    do
    for DATASET in CIFAR10 
        do

            /home/gpu2/miniconda3/envs/sunt_torch2.1.1/bin/python \
            transfer.py $DATASET $MODEL \
            -d 0 \
            -o transfer_data/$DATASET-$MODEL \
            --victim_dir victim_mia/$DATASET-$MODEL \
            --trasnferset_budgets 1000 \
            --budgets 500 \
            --argmaxed

        done
    done