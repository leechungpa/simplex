#!/bin/bash
CUDA_ID=0
BASE_DIR="./trained_models/dcl"

VALID_MODEL_DIRS=(
    # "0vapilry" "27c8rnjs" "3xiauwzx" "46rj2y0n" "560u6vda"
)

for subdir in "$BASE_DIR"/*/; do
for model_dir in "$subdir"*/; do

folder_name=$(basename "$model_dir")

if [[ " ${VALID_MODEL_DIRS[@]} " =~ " ${folder_name} " ]]; then
    # Find the files ended with '=200.ckpt'
    ckpt_file=$(find "$model_dir" -type f -name "*=200.ckpt" | head -n 1)

    if [[ -n "$ckpt_file" ]]; then
        # '=' -> '\='
        ckpt_file=$(echo "$ckpt_file" | sed 's/=/\\=/g')

        echo "Running model evaluation for: $ckpt_file"

        CUDA_VISIBLE_DEVICES=0 python3 main_linear.py --config-name linear_cifar.yaml \
            "pretrained_feature_extractor=$ckpt_file"
    fi
else
    echo "Skipping model directory: $folder_name (not in allowed list)"
fi

done
done