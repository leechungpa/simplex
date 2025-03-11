# On the Similarities of Embeddings in Contrastive Learning

This repository is forked from [vturrisi/solo-learn](https://github.com/vturrisi/solo-learn).

This repository implements **Variance-Reduction for Negative-pair similarity (VRN) loss**, which improves contrastive learning performance by reducing the variance in negative-pair similarities. The VRN loss term is designed to be integrated into self-supervised method loss functions. The repository includes scripts for self-supervised pretraining and evaluation on CIFAR and ImageNet.

## Pretraining

```bash
# Pretraining on CIFAR
CUDA_VISIBLE_DEVICES=0 python3 main_pretrain.py \
    --config-name pretrain_cifar.yaml

# Pretraining on ImageNet
CUDA_VISIBLE_DEVICES=0 python3 main_pretrain.py \
    --config-name pretrain_imagenet.yaml
```

## Linear Probing
After pretraining, perform linear probing using:

```bash
# Evaluation on CIFAR
CUDA_VISIBLE_DEVICES=0 python3 main_linear.py \
    --config-name linear_cifar.yaml

# Evaluation on ImageNet
CUDA_VISIBLE_DEVICES=0 python3 main_linear.py \
    --config-name linear_imagebet.yaml
```


## VRN Loss Implementation
The VRN loss term can be enabled in the configuration file:

```bash
add_vrn_loss_term:
  enabled: True
  weight: 30
  k: # Number of negative pairs considered 
```