run:
	python train.py\
    --dataset coco\
    --output-dir ./checkpoints/local\
    --model retinanet_resnet50_fpn --epochs 26\
    --lr-steps 16 22 --aspect-ratio-group-factor 3\
    --lr 0.01 --weights-backbone ResNet50_Weights.IMAGENET1K_V1\
    --eval_freq 3

run_test:
	python train.py\
    --dataset coco\
    --output-dir ./checkpoints/local\
    --model retinanet_resnet50_fpn --epochs 26\
    --lr-steps 16 22 --aspect-ratio-group-factor 3\
    --lr 0.01 --weights-backbone ResNet50_Weights.IMAGENET1K_V1\
    --test-only
# --resume "./checkpoints/server/checkpoint.pth"

run_server:
	sbatch --gpus=8 -p gpu_c128 run.sh 8

run_server_test:
	sbatch --gpus=8 -p gpu_c128 run.sh 8 "./checkpoints/server/checkpoint.pth"