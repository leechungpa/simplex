#!/bin/bash
CUDA_ID=0
BASE_DIR="./trained_models/dcl"

for subdir in "$BASE_DIR"/*/; do
for model_dir in "$subdir"*/; do

# Find the files ended with '=200.ckpt'
ckpt_file=$(find "$model_dir" -type f -name "*=200.ckpt" | head -n 1)

if [[ -n "$ckpt_file" ]]; then
    # '=' -> '\='
    ckpt_file=$(echo "$ckpt_file" | sed 's/=/\\=/g')

    echo "Running model evaluation for: $ckpt_file"

    CUDA_VISIBLE_DEVICES=$CUDA_ID python3 ../main_linear.py --config-name linear_cifar.yaml \
        "pretrained_feature_extractor=$ckpt_file"
fi

done
done
