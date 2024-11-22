import os
import random
import copy
from typing import List

import argparse

from prettytable import PrettyTable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.sampler import BatchSampler
import torch.nn.functional as F


import numpy as np

import matplotlib.pyplot as plt

from solo.losses.simclr import simclr_loss_func
from solo.losses.simplex import simplex_loss_func_general


############
# Utilities
def get_args_table(args_dict):
    table = PrettyTable(['Arg', 'Value'])
    for arg, val in args_dict.items():
        table.add_row([arg, val])
    return table

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    print("Seed everything: {}".format(seed))


############
# Datasets
class SyntheticDataset(Dataset):
    def __init__(self, num_samples=10000, n_class=10, std=0.5, data_dim=128, transform_matrix=None, seed=1234):
        self.seed = seed

        self.num_samples = num_samples
        self.n_class = n_class
        self.data_dim = data_dim
        self.std = std

        self.transform_matrix = transform_matrix if transform_matrix is not None else np.eye(data_dim)

        self.data, self.labels = self._generate_data()

    def _generate_data(self):
        data = []
        labels = []

        rng = np.random.default_rng(self.seed)
        for label in range(self.n_class):
            mean = np.zeros(self.data_dim, dtype=np.float32)
            mean[label] = 1.0
            # features = np.random.normal(loc=mean, scale=self.std, size=(self.num_samples // self.n_class, self.data_dim))
            features = rng.multivariate_normal(mean, np.eye(self.data_dim)*self.std, size=(self.num_samples // self.n_class))
            features = np.dot(features, self.transform_matrix.T)
            data.append(features)
            labels.extend([label] * (self.num_samples // self.n_class))
        return torch.tensor(np.vstack(data), dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class ReclassifyDataset(Dataset):
    def __init__(self, original_dataset: Dataset, label_mapping: List[int]):
        self.original_dataset = original_dataset
        self.label_mapping = label_mapping   # maps the original labels to new labels

        self.n_class = len(set(label_mapping))

    def __len__(self):
        return self.original_dataset.num_samples

    def __getitem__(self, idx):
        data, original_label = self.original_dataset[idx]
        mapped_label = self.label_mapping[original_label]
        return data, torch.tensor(mapped_label, dtype=torch.long)

class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    https://discuss.pytorch.org/t/load-the-same-number-of-data-per-class/65198/4
    https://github.com/adambielski/siamese-triplet
    """
    def __init__(self, dataset, n_classes, n_samples):
        loader = DataLoader(dataset)
        self.labels_list = []
        for _, label in loader:
            self.labels_list.append(label)
        self.labels = torch.LongTensor(self.labels_list)
        self.labels_set = list(set(self.labels.numpy()))
        
        # dictionary mapping each label to the indices of its corresponding samples in the dataset
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}

        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])

        # track the number of samples already used for each class
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples   # number of samples to draw from each class in a single batch
        self.dataset = dataset
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size <= len(self.dataset):
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(
                    self.label_to_indices[class_][
                            self.used_label_indices_count[class_]:self.used_label_indices_count[class_] + self.n_samples
                        ]
                    )
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return len(self.dataset) // self.batch_size


############
# Model
class SimpleNN(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(input_dim, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


############
# Trainer
def train_model(model, optimizer, train_loader, val_loader, test_loader, lose_type, num_epochs, args):
    acc_result = []

    if lose_type == "simplex":
        model.eval()
        ####
        # calculate the embedding properties for simplex parameter
        if args.simplex_type == "delta":
            neg_sim = []
            with torch.no_grad():
                for inputs, labels in train_loader:
                    embeddings = model(inputs).detach()
                    embeddings = F.normalize(embeddings, dim=1)
                    similarity_matrix = torch.mm(embeddings, embeddings.T)

                    target = labels.unsqueeze(0)
                    mask = target.t() == target
                    
                    neg_sim.append(similarity_matrix[~mask].mean().item())

            neg_sim = np.mean(neg_sim)
            print(f"similarity of negative pairs: {neg_sim}")
            k = 1 - 1/neg_sim
            centroid = None
        elif args.simplex_type == "centroid":
            all_labels = torch.cat([labels for _, labels in train_loader])
            k = torch.unique(all_labels).numel()   # 고유한 class 개수
            print(f"'k' of simplex loss: {k} (=number of classes)")

            centroid = torch.zeros(args.out_dim, requires_grad=False)
            n_instance = 0

            with torch.no_grad():
                for images, labels in train_loader:
                    embeddings = model(images).detach()
                    embeddings = F.normalize(embeddings, dim=1) 
                
                    centroid += embeddings.sum(axis=0)
                    n_instance += embeddings.shape[0]

                centroid = centroid / n_instance
        else:
            # 데이터셋에 포함된 class의 수
            all_labels = torch.cat([labels for _, labels in train_loader])
            n_labels_for_train_loader = torch.unique(all_labels).numel()

            
            freezed_model = copy.deepcopy(model)
            freezed_model.eval()
        ####

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            ####
            # calculate the embedding properties in batch for simplex parameter
            if lose_type == "simplex":
                if args.simplex_type == "delta_batch":
                    with torch.no_grad():
                        embeddings = freezed_model(inputs).detach()
                        embeddings = F.normalize(embeddings, dim=1)
                        similarity_matrix = torch.mm(embeddings, embeddings.T)

                        target = labels.unsqueeze(0)
                        mask = target.t() == target

                        neg_sim = similarity_matrix[~mask].mean().item()

                    print(f"similarity of negative pairs: {neg_sim}")
                    k = 1 - 1/neg_sim
                    centroid = None
                elif args.simplex_type == "centroid_batch":
                    k = n_labels_for_train_loader
                    with torch.no_grad():
                        embeddings = freezed_model(inputs).detach()
                        embeddings = F.normalize(embeddings, dim=1) 
                        centroid = embeddings.sum(axis=0) / embeddings.shape[0]
            ####

            if lose_type == "simclr":
                loss = simclr_loss_func(outputs, labels, args.simclr_t)
            elif lose_type == "simplex":
                loss = simplex_loss_func_general(
                    outputs, outputs, labels,
                    k=k, p=2, lamb=args.simplex_lamb,
                    centroid=centroid,
                    rectify_small_neg_sim=args.simplex_restrict_negative
                )
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total += labels.size(0)

        string_epoch = f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}"

        if type(val_loader) is list:
            temp_result = []
            for cnt in range(len(val_loader)):
                acc = test_nn(model, val_loader[cnt], test_loader[cnt], top_k=200)
                string_epoch += f", acc@1: {acc:.2f}"
                temp_result.append(acc)
            acc_result.append(temp_result)
        else:
            acc = test_nn(model, val_loader, test_loader, top_k=200)
            string_epoch += f", acc@1: {acc:.2f}"
            acc_result.append(acc)

        print(string_epoch)
    
    return acc_result

############
# Evaluation
def eval_sim_of_class_mean(net, data_loader, dim, n_class):
    net.eval()

    class_mean = torch.zeros(n_class, dim, requires_grad=False)
    class_n = torch.zeros(n_class, 1, requires_grad=False)

    with torch.no_grad():
        for images, labels in data_loader:
            embeddings = net(images)   # (batch_size, dim)
            for cnt_class in range(n_class):
                class_mean[cnt_class] += embeddings[labels==cnt_class].sum(axis=0)
                class_n[cnt_class] += (labels==cnt_class).sum()

        class_mean = class_mean / class_n

        class_mean = nn.functional.normalize(class_mean, dim=1)

    # class간 cosine similarity
    print(torch.mm(class_mean, class_mean.t()))

def test_nn(net, memory_data_loader, test_data_loader, top_k=200):
    net.eval()
    total_top1, total_num, feature_bank  = 0.0, 0, []

    feature_labels = []
    with torch.no_grad():
        # generate feature bank
        for data, label in memory_data_loader:
            feature = net(data)
            feat = nn.functional.normalize(feature, dim=-1)
            feature_bank.append(feat)
            feature_labels.append(label)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.cat(feature_labels, dim=0)

        n_label = torch.unique(feature_labels).shape[0]

        # loop test data to predict the label by weighted knn search
        for data, target in test_data_loader:
            feature = net(data)
            feat = nn.functional.normalize(feature, dim=-1)

            total_num += data.size(0)
            sim_matrix = torch.mm(feat, feature_bank)
            # [B, K]
            _, sim_indices = sim_matrix.topk(k=top_k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)

            # counts for each class
            one_hot_label = torch.zeros(data.size(0) * top_k, n_label, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1).long(), value=1.0)
            # score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, n_label), dim=1)
            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

    return total_top1 / total_num * 100

############
# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic data experiment.")
    
    # Model parameters
    parser.add_argument("--out_dim", type=int, default=16, help="Output dimensionality of the model.")

    # Data parameters
    parser.add_argument("--data_dim", type=int, default=32, help="Dimensionality of the input data.")

    parser.add_argument("--n_train", type=int, default=5000, help="Training sample size.")
    parser.add_argument("--n_test", type=int, default=500, help="Testing sample size.")

    parser.add_argument("--std_to_generate", type=float, default=0.3, help="Standard deviation for synthetic data generation.")

    parser.add_argument("--n_class", type=int, default=10, help="Number of classes in the dataset.")
    parser.add_argument("--linear_transform_data", action="store_true", help="Whether to apply a linear transformation to the data.")
    parser.add_argument("--normalize_data", action="store_true", help="Whether to normalize the data.")

    # Pre-training parameters
    parser.add_argument("--coarse_label", nargs="+", type=int, default=[0, 0, 1, 1, 2, 2, 3, 3, 4, 4], help="Coarse labels for pre-training.")
    parser.add_argument("--batchsize", type=int, default=1000, help="Batch size for pre-training.")
    parser.add_argument("--epoch_pretrain", type=int, default=100, help="Number of pre-training epochs.")

    # Fine-tuning parameters
    parser.add_argument("--fine_labels", nargs="+", type=int, default=[0, 1, 2, 3], help="Labels for fine-tuning.")
    parser.add_argument("--batchsize_finetune", type=int, default=32, help="Batch size for fine-tuning.")
    parser.add_argument("--epoch_finetune", type=int, default=30, help="Number of fine-tuning epochs.")

    # Hyper parameters
    parser.add_argument("--seed", type=int, default=1234, help="Seed")

    parser.add_argument("--simclr_t", type=float, default=0.5, help="Temperature parameter for SimCLR loss")
    parser.add_argument("--simplex_lamb", type=float, default=1.0, help="Lambda parameter for simplex loss.")
    parser.add_argument("--simplex_use_centroid", action="store_true", help="Using centroid for simplex loss.")
    parser.add_argument("--simplex_type", type=str, default="centroid", help="Simplex loss type.")
    parser.add_argument("--simplex_restrict_negative", action="store_true", help="Ensure that negative pairs move farther apart.")

    parser.add_argument("--lr_pretrain_simclr", type=float, default=1.0, help="Learning rate for SimCLR pre-training.")
    parser.add_argument("--lr_simclr", type=float, default=0.05, help="Learning rate for SimCLR fine-tuning.")
    parser.add_argument("--lr_simplex", type=float, default=0.1, help="Learning rate for Simplex fine-tuning.")

    # Output
    parser.add_argument("--output_name", type=str, default="./synthetic_result.png", help="Path to save the output plot.")
    parser.add_argument("--output_pretrain", type=str, default="./syn_result/pretrained_model.pt", help="Path to save the output plot.")

    # ETC
    print_eval_sim_of_class_mean = False
    torch.set_printoptions(precision=2, sci_mode=False)

    args = parser.parse_args()
    print(get_args_table(vars(args)))

    ##########
    # Pre training
    set_seed(args.seed)
    if args.linear_transform_data:
        transformation_matrix = np.random.rand(args.data_dim, args.data_dim)  # Random matrix for transformation
    else:
        transformation_matrix = None

    train_dataset = SyntheticDataset(args.n_train, args.n_class, args.std_to_generate, args.data_dim, transformation_matrix, args.seed)
    test_dataset = SyntheticDataset(args.n_test, args.n_class, args.std_to_generate, args.data_dim, transformation_matrix, args.seed)

    if args.normalize_data:
        mean = train_dataset.data.mean(axis=0)
        std = train_dataset.data.std(axis=0)

        train_dataset.data = (train_dataset.data - mean) / std
        test_dataset.data = (test_dataset.data - mean) / std

    coarse_train_dataset = ReclassifyDataset(train_dataset, args.coarse_label)
    coarse_test_dataset = ReclassifyDataset(test_dataset, args.coarse_label)
    coarse_n_class = len(set(args.coarse_label))

    coarse_balanced_sampler = BalancedBatchSampler(coarse_train_dataset, coarse_n_class, args.batchsize//coarse_n_class)

    coarse_train_loader = DataLoader(coarse_train_dataset, batch_sampler=coarse_balanced_sampler)
    coarse_val_loader = DataLoader(coarse_train_dataset, batch_size=args.batchsize, shuffle=False)
    coarse_test_loader = DataLoader(coarse_test_dataset, batch_size=args.batchsize, shuffle=False)

    model = SimpleNN(input_dim=args.data_dim, out_dim=args.out_dim)
    optimizer = optim.SGD(model.parameters(), lr=args.lr_pretrain_simclr, weight_decay=1e-6)

    if os.path.exists(args.output_pretrain):
        model.load_state_dict(torch.load(args.output_pretrain, weights_only=False))
    else:
        train_model(
            model, optimizer,
            coarse_train_loader, coarse_val_loader, coarse_test_loader,
            "simclr", args.epoch_pretrain, args
        )
        torch.save(model.state_dict(), args.output_pretrain)

    if print_eval_sim_of_class_mean:
        eval_sim_of_class_mean(model, coarse_test_loader, args.out_dim, coarse_n_class)
        eval_sim_of_class_mean(model, test_dataset, args.out_dim, args.n_class)

    ##########
    # Finetuning
    fine_n_class = len(set(args.fine_labels))
    indices = [idx for idx, label in enumerate(train_dataset.labels) if label in args.fine_labels]
    fine_train_dataset = Subset(train_dataset, indices)
    fine_balanced_sampler = BalancedBatchSampler(fine_train_dataset, fine_n_class, args.batchsize_finetune//fine_n_class)

    indices = [idx for idx, label in enumerate(test_dataset.labels) if label in args.fine_labels]
    fine_test_dataset = Subset(test_dataset, indices)

    fine_train_loader = DataLoader(fine_train_dataset, batch_sampler=fine_balanced_sampler)
    fine_val_loader = DataLoader(fine_train_dataset, batch_size=args.batchsize_finetune, shuffle=False)
    fine_test_loader = DataLoader(fine_test_dataset, batch_size=args.batchsize_finetune, shuffle=False)

    acc_finetune = test_nn(model, fine_val_loader, fine_test_loader, top_k=200)
    acc_coarse = test_nn(model, coarse_val_loader, coarse_test_loader, top_k=200)
    print(f"Before fine-tuning: {acc_finetune:.2f}, {acc_coarse:.2f}")
    
    pretrained_model_simclr = copy.deepcopy(model)
    pretrained_model_simplex = copy.deepcopy(model)

    print("----simclr----")
    finetune_optimizer_simclr = optim.SGD(pretrained_model_simclr.parameters(), lr=args.lr_simclr)
    simclr_result = train_model(
        pretrained_model_simclr, finetune_optimizer_simclr,
        fine_train_loader, [fine_val_loader, coarse_val_loader], [fine_test_loader, coarse_test_loader],
        "simclr", args.epoch_finetune, args=args
    )

    if print_eval_sim_of_class_mean:
        eval_sim_of_class_mean(pretrained_model_simclr, coarse_test_loader, args.out_dim, coarse_n_class)
        eval_sim_of_class_mean(pretrained_model_simclr, fine_test_dataset, args.out_dim, fine_n_class)
        eval_sim_of_class_mean(pretrained_model_simclr, test_dataset, args.out_dim, args.n_class)

    print("----simplex----")
    finetune_optimizer_simplex = optim.SGD(pretrained_model_simplex.parameters(), lr=args.lr_simplex)
    simplex_result = train_model(
        pretrained_model_simplex, finetune_optimizer_simplex,
        fine_train_loader, [fine_val_loader, coarse_val_loader], [fine_test_loader, coarse_test_loader],
        "simplex", args.epoch_finetune, args=args
    )

    if print_eval_sim_of_class_mean:
        eval_sim_of_class_mean(pretrained_model_simplex, coarse_test_loader, args.out_dim, coarse_n_class)
        eval_sim_of_class_mean(pretrained_model_simplex, fine_test_dataset, args.out_dim, fine_n_class)
        eval_sim_of_class_mean(pretrained_model_simplex, test_dataset, args.out_dim, args.n_class)

    ##########
    # Plot the result
    plt.figure(figsize=(10, 6))
    simclr_x, simclr_y = zip(*simclr_result)
    simplex_x, simplex_y = zip(*simplex_result)

    plt.scatter(acc_finetune, acc_coarse, label='Pre-trained base model', color='black', s=100)

    plt.scatter(simclr_x, simclr_y, label='Fine-tuning using SimCLR', color='blue')
    plt.scatter(simplex_x, simplex_y, label='Fine-tuning using Simplex', color='red')
    
    for i, (x, y) in enumerate(zip(simclr_x, simclr_y), start=1):
        plt.annotate(str(i), (x, y), textcoords="offset points", xytext=(5, 5), ha='center', color='blue')

    for i, (x, y) in enumerate(zip(simplex_x, simplex_y), start=1):
        plt.annotate(str(i), (x, y), textcoords="offset points", xytext=(5, 5), ha='center', color='red')


    plt.xlabel('Acc@1 (fine-tune task: subset of fine-grained classes)')
    plt.ylabel('Acc@1 (pre-train task: coarse-grained classes)')
    plt.legend(loc='lower left')
    plt.grid()

    plt.savefig(args.output_name)
    plt.close()
