export PYTHONPATH=../:$PYTHONPATH

for MODEL in resnet18 
    do
    for DATASET in CIFAR10 
        do
        /home/gpu2/miniconda3/envs/sunt_torch2.1.1/bin/python3 \
        train_victim.py $DATASET $MODEL \
        -d 0 \
        -o victim/$DATASET-$MODEL \
        -e 100 \
        --lr 1e-2 \
        --pretrained 
    done
done
