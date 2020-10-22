import torch
import torchvision
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data.sampler import SubsetRandomSampler

def load_data(dataset, valid_size):
    if dataset == 'cifar10':
        num_classes = 10

        transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        ### trainset
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform_train)
        valset = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                                 transform=transform_test, download=False)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform_test)
        
        indices = torch.randperm(len(trainset))
        train_indices = indices[:len(indices) - valid_size]
        valid_indices = indices[len(indices) - valid_size:] if valid_size else None

    elif dataset == 'cifar100':
        num_classes = 100

        transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        ### trainset
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                                download=True, transform=transform_train)
        valset = torchvision.datasets.CIFAR100(root='./data', train=True, 
                                                 transform=transform_test, download=False)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                               download=True, transform=transform_test)
        
        indices = torch.randperm(len(trainset))
        train_indices = indices[:len(indices) - valid_size]
        valid_indices = indices[len(indices) - valid_size:] if valid_size else None

    elif dataset == 'tinyimagenet':
        num_classes = 200

        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        transform_test = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        dataroot = '/home/zz452/data/tiny-imagenet-200'
        train_val_dataset_dir = os.path.join(dataroot, "train")
        test_dataset_dir = os.path.join(dataroot, "val1")

        trainset = datasets.ImageFolder(root=train_val_dataset_dir, transform=transform_train)
        valset = datasets.ImageFolder(root=train_val_dataset_dir, transform=transform_test)
        testset  = datasets.ImageFolder(root=test_dataset_dir, transform=transform_test)

        indices = torch.randperm(len(trainset))
        train_indices = indices[:len(indices) - 10000]
        valid_indices = indices[len(indices) - 10000:]

    else:
        print('dataset not implemented!')

    # Make dataloaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               sampler=SubsetRandomSampler(train_indices), num_workers=2)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                               sampler=SubsetRandomSampler(valid_indices), num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, drop_last = False, num_workers=2) 
    
    return trainloader, valloader, testloader