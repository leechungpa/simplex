# On the Similarities of Embeddings in Contrastive Learning

This repository is forked from [vturrisi/solo-learn](https://github.com/vturrisi/solo-learn).

This repository implements **Variance-Reduction for Negative-pair similarity (VRN) loss**, which improves contrastive learning performance by reducing the variance in negative-pair similarities. The VRN loss term is designed to be integrated into self-supervised method loss functions. The repository includes scripts for self-supervised pretraining and evaluation on CIFAR and ImageNet.

## Pretraining

### Pretraining on CIFAR

```bash
CUDA_VISIBLE_DEVICES=0 python3 main_pretrain.py \
    --config-path # --- path to config directory --- \
    --config-name cifar.yaml
```


### Pretraining on ImageNet

```bash
CUDA_VISIBLE_DEVICES=0 python3 main_pretrain.py \
    --config-path # --- path to config directory --- \
    --config-name imagenet.yaml
```

## Linear Evaluation
After pretraining, perform linear evaluation using:

```bash
CUDA_VISIBLE_DEVICES=0 python3 main_linear.py \
    --config-path # --- path to config directory --- \
    --config-name _eval_linear.yaml
```


## VRN Loss Implementation
The VRN loss term can be enabled in the configuration file:

```bash
add_vrn_loss_term:
  enabled: True
  weight: 30
  p: 2  
  k: # Number of negative pairs considered 
```