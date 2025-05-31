export PYTHONPATH=../:$PYTHONPATH

for MODEL in  resnet50
    do
    for DATASET in stl10
        do
        /home/gpu2/miniconda3/envs/sunt_torch2.1.1/bin/python3 \
        train_victim.py $DATASET $MODEL \
        -d 0 \
        -o victim_mia/$DATASET-$MODEL \
        -e 100 \
        --lr 1e-2 \
        --pretrained

        /home/gpu2/miniconda3/envs/sunt_torch2.1.1/bin/python3 \
        transfer.py $DATASET $MODEL \
        -d 0 \
        -o transfer_data/$DATASET-$MODEL \
        --victim_dir victim_mia/$DATASET-$MODEL \
        --trasnferset_budgets 100 \
        --budgets 100 \
        --argmaxed

        /home/gpu2/miniconda3/envs/sunt_torch2.1.1/bin/python3 \
        train_layer.py $DATASET $MODEL \
        -d 0 \
        -o results/train_layer/$DATASET-$MODEL \
        --transfer_path  transfer_data/$DATASET-$MODEL \
        --victim_dir victim_mia/$DATASET-$MODEL \
        --shadow_model_dir shadow_mia/$DATASET-$MODEL \
        --budgets 100 \
        --lr 0.1 \
        --remain-lr 1e-3 \
        --update-lr 1e-2 \
        --epochs 10 \
        --pretrained \
        --graybox-mode block_shallow \
        --argmaxed 

        /home/gpu2/miniconda3/envs/sunt_torch2.1.1/bin/python3 \
        train_layer.py $DATASET $MODEL \
        -d 0 \
        -o results/train_layer_deep/$DATASET-$MODEL \
        --transfer_path  transfer_data/$DATASET-$MODEL \
        --victim_dir victim_mia/$DATASET-$MODEL \
        --shadow_model_dir shadow_mia/$DATASET-$MODEL \
        --budgets 100 \
        --lr 0.1 \
        --remain-lr 1e-3 \
        --update-lr 1e-2 \
        --epochs 10 \
        --pretrained \
        --graybox-mode block_deep \
        --argmaxed 

        # /home/gpu2/miniconda3/envs/sunt_torch2.1.1/bin/python3 \
        # rm_subdir_meminf.py results/train_layer/$DATASET-$MODEL


    done
done
