import os
import random
import warnings

import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100, CIFAR10, Caltech101, Flowers102
from torch.utils.data import ConcatDataset, random_split, Subset

from solo.data.downstream.cub200 import CUB
from solo.data.downstream.dogs import Dogs
from solo.data.downstream.datasets import Pets, Food101, DTD, SUN397, MIT67


def setup_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class logger(object):
    def __init__(self, path="log.txt", local_rank=0):
        self.path = path
        self.local_rank = local_rank

    def info(self, msg):
        if self.local_rank == 0:
            print(msg)
            with open(self.path, 'a') as f:
                f.write(msg + "\n")

def get_transfer_dataset(args):
    transform = transforms.Compose([transforms.Resize(224, interpolation=transforms.functional.InterpolationMode.BICUBIC), 
                                    transforms.CenterCrop(224),transforms.ToTensor(), 
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    generator = lambda seed: torch.Generator().manual_seed(seed)
    if args.data == 'food101':
        trainval = Food101(root=os.path.join(args.data_dir, 'food'), split='train', transform=transform)
        train_data, val_data = random_split(trainval, [68175, 7575], generator=generator(42))
        test_data = Food101(root=os.path.join(args.data_dir, 'food'), split='test',  transform=transform)
        num_classes = 101
        
    elif args.data =='CIFAR10':
        trainval = CIFAR10(root=args.data_dir, train=True, transform=transform)
        train_data, val_data = random_split(trainval, [45000, 5000], generator=generator(43))
        test_data = CIFAR10(root=args.data_dir, train=False, transform=transform)
        num_classes = 10
        
    elif args.data == 'CIFAR100':
        trainval = CIFAR100(root=args.data_dir, train=True, transform=transform)
        train_data, val_data = random_split(trainval, [45000, 5000], generator=generator(44))
        test_data = CIFAR100(root=args.data_dir, train=False, transform=transform)
        num_classes = 100

    elif args.data == 'sun397':
        trn_indices, val_indices = torch.load('./solo/data/downstream/split/sun397.pth')
        trainval = SUN397(root = os.path.join(args.data_dir, 'SUN397'), split='Training', transform=transform)
        train_data = Subset(trainval, trn_indices)
        val_data   = Subset(trainval, val_indices)
        test_data = SUN397(root = os.path.join(args.data_dir, 'SUN397'), split='Testing', transform=transform)
        num_classes = 397

    elif args.data == 'dtd':
        train_data = DTD(root=os.path.join(args.data_dir, 'dtd'), split='train', transform=transform)
        val_data = DTD(root=os.path.join(args.data_dir, 'dtd'), split='val',   transform=transform)
        trainval = ConcatDataset([train_data, val_data])
        test_data = DTD(root=os.path.join(args.data_dir, 'dtd'), split='test',  transform=transform)
        num_classes = 47

    elif args.data == 'pets':
        trainval = Pets(root=os.path.join(args.data_dir, 'pets'), split='trainval', transform=transform)
        train_data, val_data = random_split(trainval, [2940, 740], generator=generator(49))
        test_data = Pets(root=os.path.join(args.data_dir, 'pets'), split='test', transform=transform)
        num_classes = 37

    elif args.data == 'caltech101':
        transform.transforms.insert(0, transforms.Lambda(lambda img: img.convert('RGB')))
        D = Caltech101(root=args.data_dir, transform=transform)
        trn_indices, val_indices, tst_indices = torch.load('./solo/data/downstream/split/caltech101.pth')
        train_data = Subset(D, trn_indices)
        val_data = Subset(D, val_indices)
        trainval = ConcatDataset([train_data, val_data])
        test_data= Subset(D, tst_indices)
        num_classes = 101

    elif args.data == 'flowers102':
        train_data = Flowers102(root = args.data_dir, split='train', transform=transform)
        val_data = Flowers102(root = args.data_dir, split='val', transform=transform)
        trainval = ConcatDataset([train_data, val_data])
        test_data = Flowers102(root = args.data_dir, split='test', transform=transform)
        num_classes = 102

    elif args.data == 'mit67':
        trainval = MIT67(root = os.path.join(args.data_dir, 'mit67'), split='Train', transform=transform)
        test_data = MIT67(root = os.path.join(args.data_dir, 'mit67'), split='Test', transform=transform)
        train_data, val_data = random_split(trainval, [4690, 670], generator=generator(51))
        num_classes = 67

    elif args.data == 'cub200':
        transform.transforms.insert(0, transforms.Lambda(lambda img: img.convert('RGB')))
        train_data = CUB(os.path.join(args.data_dir, 'cub200'), 'train', transform=transform)
        val_data = CUB(os.path.join(args.data_dir, 'cub200'), 'valid', transform=transform)
        trainval = ConcatDataset([train_data, val_data])
        test_data = CUB(os.path.join(args.data_dir, 'cub200'), 'test', transform=transform)
        num_classes = 200

    elif args.data == 'dog':
        trn_indices, val_indices = torch.load('./solo/data/downstream/split/dog.pth')
        trainval = Dogs(args.data_dir, train=True, transform=transform)
        train_data = Subset(trainval, trn_indices)
        val_data = Subset(trainval, val_indices)
        test_data = Dogs(args.data_dir, train=False, transform = transform)
        num_classes = 120
        
    else:  
        warnings.warn('You have chosen wrong data')
        
    return trainval, train_data, val_data, test_data, num_classes

