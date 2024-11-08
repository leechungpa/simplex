import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
from tqdm import tqdm
from torch.utils.data.sampler import BatchSampler

import torch.nn.functional as F

import copy

from solo.losses.simclr import simclr_loss_func
from solo.losses.simplex import simplex_loss_func

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



class SyntheticDataset(Dataset):
    def __init__(self, num_samples=10000, n_class=10, std=0.5, input_dim=128, affine_matrix=None, affine_bias=None):
        self.num_samples = num_samples
        self.n_class = n_class
        self.input_dim = input_dim

        self.std = std

        self.affine_matrix = affine_matrix if affine_matrix is not None else np.eye(input_dim)
        self.affine_bias = affine_bias if affine_bias is not None else np.zeros(input_dim)

        self.data, self.labels = self._generate_data()

    def _generate_data(self):
        data = []
        labels = []
        for label in range(self.n_class):
            mean = np.zeros(self.input_dim, dtype=np.float32)
            mean[label] = 1.0
            features = np.random.normal(loc=mean, scale=self.std, size=(self.num_samples // self.n_class, self.input_dim))
            features = np.dot(features, self.affine_matrix.T) + self.affine_bias
            data.append(features)
            labels.extend([label] * (self.num_samples // self.n_class))
        return torch.tensor(np.vstack(data), dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]



class LabelMappedDataset(Dataset):
    def __init__(self, original_dataset, label_mapping):
        self.original_dataset = original_dataset
        self.label_mapping = label_mapping

        self.n_class = len(set(label_mapping))

    def __len__(self):
        return self.original_dataset.num_samples

    def __getitem__(self, idx):
        data, original_label = self.original_dataset[idx]
        mapped_label = self.label_mapping[original_label]
        return data, torch.tensor(mapped_label, dtype=torch.long)
    


class SubsetByLabels(Dataset):
    def __init__(self, original_dataset, allowed_labels):
        self.original_dataset = original_dataset
        self.allowed_labels = set(allowed_labels)

        self.filtered_indices = [
            idx for idx, (_, label) in enumerate(original_dataset)
            if label in self.allowed_labels
        ]

        self.n_class = len(set(allowed_labels))

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, idx):
        original_idx = self.filtered_indices[idx]
        return self.original_dataset[original_idx]




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
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
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


def train_model(model, optimizer, train_loader, val_loader, test_loader, lose_type, num_epochs=10):
    acc_result = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            if lose_type == "simclr":
                loss = simclr_loss_func(outputs, labels, 0.1)
            elif lose_type == "simplex":

                with torch.no_grad():
                    normalized_outputs = F.normalize(outputs, dim=1) # ( B, proj_output_dim )
                    similarity_matrix = torch.mm(normalized_outputs, normalized_outputs.T)   # (N, N)
                    
                    target = labels.unsqueeze(0)
                    mask = target.t() == target
                    
                    negative_similarity = similarity_matrix[~mask].view(similarity_matrix.size(0), -1)   # (N, N-1)
                    avg_neg_similarity = negative_similarity.mean().item()

                    print(avg_neg_similarity)
                    print(similarity_matrix[mask].mean())

                    k = 1 - 1/avg_neg_similarity

                loss = simplex_loss_func(outputs, outputs, labels, k=k, p=2, lamb=20)
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
        print(string_epoch)
        acc_result.append(acc)



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
            # data, target = data.cuda(args.gpu, non_blocking=True), target.cuda(args.gpu, non_blocking=True)
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


if __name__ == "__main__":
    ##########
    # Pre training
    input_dim = 32
    out_dim = 32

    std_to_generate = 0.8

    n_class = 10
    
    batch_size = 100

    set_seed(1234)
    affine_matrix = np.random.rand(input_dim, input_dim)  # Random matrix for affine transformation
    affine_bias = np.random.rand(input_dim) 

    train_dataset = SyntheticDataset(5000, n_class, std_to_generate, input_dim, affine_matrix, affine_bias)
    test_dataset = SyntheticDataset(500, n_class, std_to_generate, input_dim, affine_matrix, affine_bias)

    mean = train_dataset.data.mean(axis=0)
    std = train_dataset.data.std(axis=0)

    train_dataset.data = (train_dataset.data - mean) / std
    test_dataset.data = (test_dataset.data - mean) / std


    coarse_train_dataset = LabelMappedDataset(train_dataset, [0,0,1,1,2,2,3,3,4,4])
    coarse_test_dataset = LabelMappedDataset(test_dataset, [0,0,1,1,2,2,3,3,4,4])
    coarse_n_class = 5

    coarse_balanced_sampler = BalancedBatchSampler(coarse_train_dataset, coarse_n_class, batch_size//coarse_n_class)

    coarse_train_loader = DataLoader(coarse_train_dataset, batch_sampler=coarse_balanced_sampler)
    coarse_val_loader = DataLoader(coarse_train_dataset, batch_size=batch_size, shuffle=False)
    coarse_test_loader = DataLoader(coarse_test_dataset, batch_size=batch_size, shuffle=False)


    model = SimpleNN(input_dim=input_dim, out_dim=out_dim)
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    train_model(model, optimizer, coarse_train_loader, coarse_val_loader, coarse_test_loader, "simclr", num_epochs=20)

    ##########
    # Finetuning
    labels = [0,1,2,3]
    fine_n_class = 4

    batch_size = 32

    epoch_finetue = 10

    indices = [idx for idx, label in enumerate(train_dataset.labels) if label in labels]
    fine_train_dataset = Subset(train_dataset, indices)
    fine_balanced_sampler = BalancedBatchSampler(fine_train_dataset, fine_n_class, batch_size//fine_n_class)

    indices = [idx for idx, label in enumerate(test_dataset.labels) if label in labels]
    fine_test_dataset = Subset(test_dataset, indices)

    fine_train_loader = DataLoader(fine_train_dataset, batch_sampler=fine_balanced_sampler)
    fine_val_loader = DataLoader(fine_train_dataset, batch_size=batch_size, shuffle=False)
    fine_test_loader = DataLoader(fine_test_dataset, batch_size=batch_size, shuffle=False)

    acc = test_nn(model, fine_val_loader, fine_test_loader, top_k=200)
    print(f"Before fine-tuning: {acc:.2f}")

    
    pretrained_model = copy.deepcopy(model)
    finetune_optimizer = optim.SGD(pretrained_model.parameters(), lr=0.1)

    print("----simclr----")
    train_model(pretrained_model, finetune_optimizer, fine_train_loader, [fine_val_loader, coarse_val_loader], [fine_test_loader, coarse_test_loader], "simclr", num_epochs=epoch_finetue)


    pretrained_model = copy.deepcopy(model)
    finetune_optimizer = optim.SGD(pretrained_model.parameters(), lr=0.1)

    print("----simplex----")
    train_model(pretrained_model, finetune_optimizer, fine_train_loader, [fine_val_loader, coarse_val_loader], [fine_test_loader, coarse_test_loader], "simplex", num_epochs=epoch_finetue)