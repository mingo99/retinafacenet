#!/bin/bash
date 
tar -xf ../../data/cocofb.tar -C /dev/shm/
date

module load anaconda/2021.05
source activate pytorch
export PYTHONUNBUFFERED=1

if [ -n "$2" ]; then
    torchrun --nproc_per_node=$1 train.py\
    --dataset coco\
    --data-path /dev/shm/cocofb\
    --output-dir ./checkpoints/server\
    --model retinafacenet_resnet50_fpn --epochs 26\
    --lr-steps 16 22 --aspect-ratio-group-factor 3\
    --lr 0.01 --weights-backbone ResNet50_Weights.IMAGENET1K_V1\
    --eval_freq 3\
    --test-only\
    --resume $2
else
    torchrun --nproc_per_node=$1 train.py\
    --dataset coco\
    --data-path /dev/shm/cocofb\
    --output-dir ./checkpoints/server\
    --model retinafacenet_resnet50_fpn --epochs 26\
    --lr-steps 16 22 --aspect-ratio-group-factor 3\
    --lr 0.01 --weights-backbone ResNet50_Weights.IMAGENET1K_V1\
    --eval_freq 3
fi
    
