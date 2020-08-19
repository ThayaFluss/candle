
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

from candle.io.arguments import parser_init, parse_device
from candle.io.arguments import write_config, read_config, log_dir



def main(args):
    device = parse_device(args)
    DATASET = CIFAR10
    args.jobname =  "train_resnet_{}".format(DATASET.__name__)
    dirname = log_dir(args.jobname) ### 
    net_pt = "{}/net.pt".format(dirname)

    net = ResNet(num_classes=10)
    args.net = "ResNet"
    write_config(dirname)
    temp_dict = read_config(dirname) ## for testing read & write
    
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
    num_epoch = args.epoch
    lr = args.lr
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    #optimizer = torch.optim.Adam(net.parameters(), lr=lr) 
    lambda1 = lambda epoch: epoch // 20
    lambda2 = lambda epoch: 0.95 ** epoch
    scheduler = LambdaLR(optimizer, lambda2 )
     

    for _ in tqdm(range(num_epoch)):
        #print("epoch:{}".format(_))
        train(train_loader, net, optimizer, device=device, log_dir=dirname)
        scheduler.step()
        test(test_loader, net, device=device, log_dir=dirname, debug=True)

    torch.save(net.state_dict(), net_pt)



if __name__ == "__main__":
    #args  = argsp()
    parser = parser_init()
    args = parser.parse_args([
        "--epoch", "50",
        "--batch", "100",
        "--lr", "2e-2"
                        ])

    print(args)
    main(args)