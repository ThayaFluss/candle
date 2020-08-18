from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import torch
from tqdm import tqdm


from candle.datasets.cifar10 import CIFAR10T
from candle.tpl.util import  train, test
from candle.models.resnet import ResNet


def main():
    device = "cuda"
    DATASET = CIFAR10
    task_name = "train_resnet_{}".format(DATASET.__name__)
    dirname = "log/{}".format(task_name)
    net_pt = "weights/{}_net.pt".format(task_name)
    os.makedirs(net_pt)
    net = ResNet(num_classes=10)
    net.to(device)

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    train_set = DATASET(root='./data/cifar10', 
                        train=True,
                        download=True,
                        transform=transform,
                        target_transform=None)
    test_set = DATASET(root='./data/cifar10', 
                        train=False,
                        download=True, 
                        transform=transform,
                        target_transform=None)


    num_workers=4
    test_loader = DataLoader(test_set, 
                        batch_size=500,
                        shuffle=False, 
                        num_workers=num_workers,
                        drop_last=False)


    train_loader = DataLoader(test_set, 
                        batch_size=100,
                        shuffle=True, 
                        num_workers=num_workers,
                        drop_last=False)

    test(test_loader, net, device=device, log_dir=dirname, debug=True)
    ### pretrain
    print("Pretraining...")
    num_epoch = 10
    for _ in range(num_epoch):
        print("epoch:{}".format(_))
        net.to(device)
        lr = 2e-2
        if _ > 4:
            lr = 2e-3
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
        #optimizer = torch.optim.Adam(net.parameters(), lr=2e-2 ) 
        train(leak_train_loader, net, optimizer, device=device, log_dir=dirname
        #,max_iter=50
        )
        test(test_loader, net, device=device, log_dir=dirname, debug=True)
    torch.save(net.state_dict(), net_pt)
