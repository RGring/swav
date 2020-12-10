#!/bin/bash
DATASET_PATH="/home/ronjag/data/deepweeds"
EXPERIMENT_PATH="/home/ronjag/data/deepweeds/log/test_script"

python3 -m torch.distributed.launch --master_port=$RANDOM --nproc_per_node=1 main_swav.py \
--data_path $DATASET_PATH/train \
--dump_path $EXPERIMENT_PATH/ss \
--nmb_crops 2 6 \
--size_crops 128 96 \
--min_scale_crops 0.25 0.2 \
--max_scale_crops 1. 0.35 \
--crops_for_assign 0 1 \
--temperature 0.1 \
--epsilon 0.05 \
--sinkhorn_iterations 3 \
--feat_dim 128 \
--nmb_prototypes 3000 \
--queue_length 0 \
--epoch_queue_starts 0 \
--epochs 1 \
--batch_size 32 \
--base_lr 0.01 \
--final_lr 0.0001 \
--freeze_prototypes_niters 313 \
--wd 0.000001 \
--warmup_epochs 10 \
--start_warmup 0.1 \
--arch resnet34 \
--hidden_mlp 256 \
--workers 4 \
--aug_rotation_prob 0.0 \
--aug_jitter_vector 0.48 0.48 0.48 0.12 \
--aug_rgb_shift_prob 0.0 \
--aug_grey_prob 0.2 \
--aug_gaussian_blur_prob 0.0 \

python3 -m torch.distributed.launch --master_port=$RANDOM --nproc_per_node=1 eval_semisup_custom.py \
--data_path $DATASET_PATH \
--workers 4 \
--batch_size 32 \
--dump_path $EXPERIMENT_PATH/ft_01 \
--pretrained $EXPERIMENT_PATH/ss/checkpoint.pth.tar \
--epochs 1 \
--decay_epochs 10 15 \
--n_classes 9 \
--labels_perc 0.1 \

python3 -m torch.distributed.launch --master_port=$RANDOM --nproc_per_node=1 eval_semisup_custom.py \
--data_path $DATASET_PATH \
--workers 4 \
--batch_size 32 \
--dump_path $EXPERIMENT_PATH/ft_03 \
--pretrained $EXPERIMENT_PATH/ss/checkpoint.pth.tar \
--epochs 1 \
--decay_epochs 10 15 \
--n_classes 9 \
--labels_perc 0.3 \
