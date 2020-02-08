import torch
import torchvision
import torchvision.transforms as transforms


mean_cifar = [x / 255 for x in [125.3, 123.0, 113.9]]
std_cifar = [x / 255 for x in [63.0, 62.1, 66.7]]


def get_CIFAR(data_augmentation=False, batch_size=128, num_workers=16, mean=mean_cifar, std=std_cifar):

    test_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean, std)])

    if data_augmentation:
        transform_train = transforms.Compose(
            [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(),
             transforms.Normalize(mean_cifar, std_cifar)])
    else:
        transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                              drop_last=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                             drop_last=True)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, mean, std, classes

