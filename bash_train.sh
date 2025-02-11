# #!/bin/bash
CUDA_ID=0
MAX_EPOCHS=200
METHOD="simclr" # dcl dhel simclr

for batch_size in 512 256 128 64 32; do
for temperature in 0.1 0.2 0.3 0.4 0.5; do
for weight in 30 300 3000; do # 0 1 10 100 1000

NAME="${METHOD}_bs${batch_size}_t${temperature}_w${weight}"
echo "Running: $NAME"

CUDA_VISIBLE_DEVICES=$CUDA_ID python3 main_pretrain.py --config-name main_cifar.yaml \
    method=$METHOD \
    optimizer.batch_size=$batch_size max_epochs=$MAX_EPOCHS \
    method_kwargs.temperature=$temperature add_simplex_loss.weight=$weight \
    name=$NAME

done
done
done
