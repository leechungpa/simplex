python3 main_knn.py \
    --dataset cifar10 \
    --train_data_path ./datasets \
    --val_data_path ./datasets \
    --batch_size 256 \
    --num_workers 1 \
    --pretrained_checkpoint_dir trained_models/barlow_twins/n04u9hrp \
    --k 1 \
    --temperature 0.01 0.02 0.05 0.07 0.1 0.2 0.5 1 \
    --feature_type backbone projector \
    --distance_function euclidean cosine