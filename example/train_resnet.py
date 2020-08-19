
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import LambdaLR
import torch
from tqdm import tqdm

from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms


import os, sys
sys.path.append( os.path.dirname(__file__) + "/../" )


#from candle.datasets.cifar10 import CIFAR10T
from candle.tpl.util import  train, test
from candle.models.resnet import ResNet



def main():
    device = "cuda"
    DATASET = CIFAR10
    task_name = "train_resnet_{}".format(DATASET.__name__)
    os.makedirs("log", exist_ok=True)
    dirname = "log/{}".format(task_name)
    os.makedirs("weights", exist_ok=True)
    net_pt = "weights/{}_net.pt".format(task_name)
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
    train_loader = DataLoader(train_set, 
                        batch_size=100,
                        shuffle=True, 
                        num_workers=num_workers,
                        drop_last=False)

    test_loader = DataLoader(test_set, 
                        batch_size=500,
                        shuffle=False, 
                        num_workers=num_workers,
                        drop_last=False)



    test(test_loader, net, device=device, log_dir=dirname, debug=True)
    
    ### pretrain
    print("Pretraining...")
    num_epoch = 50
    lr = 2e-2
    #optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr) 
    lambda1 = lambda epoch: epoch // 20
    lambda2 = lambda epoch: 0.95 ** epoch
    scheduler = LambdaLR(optimizer, lambda2 )
     

    for _ in tqdm(num_epoch):
        #print("epoch:{}".format(_))
        train(train_loader, net, optimizer, device=device, log_dir=dirname)
        scheduler.step()
        test(test_loader, net, device=device, log_dir=dirname, debug=True)

    torch.save(net.state_dict(), net_pt)



if __name__ == "__main__":
    main()