import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, FashionMNIST, SVHN
import torch.optim as optim
from tqdm import tqdm

import matplotlib.pyplot as plt
import os

import scipy as sp
import math

from .plot_util import touch, plotter
#from .plot_util import plot_spectrum, plot_empirical_capacity, plot_mean,plot_spectrum_ntk
from .plot_util import get_plot_dir, write_config
from .util import get_args_of_current_function
#from .models.mlp import MLP


import time

def train_test_set_conv(DATASET=SVHN, shape=(3,32,32)):
    transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Lambda(lambda x : x.flatten() ), 
        transforms.Lambda(lambda x : (x - torch.mean(x))/ torch.sqrt (torch.mean(x**2) )),
        transforms.Lambda(lambda x : x.reshape(shape)) 
        ])

    if DATASET.__name__ == "SVHN":
            train_set = DATASET(root='./data', 
                                split="train",
                                download=True,
                                transform=transform,
                                target_transform=None)
            test_set = DATASET(root='./data', 
                                split="test", 
                                download=True, 
                                transform=transform,
                                target_transform=None)

    elif DATASET.__name__ == "CIFAR10T":
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
    else:
        raise ValueError()


    return train_set, test_set

def train_test_loader_conv(mean=0., std=1, batch_size=100, test_batch_size=500, num_workers=3, DATASET=SVHN, \
    do_target_transform=False):
    train_set, test_set = train_test_set_conv(DATASET=DATASET)
    train_loader = torch.utils.data.DataLoader(train_set,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_set, 
                                                batch_size=test_batch_size,
                                                shuffle=False, 
                                                num_workers=num_workers)

    return train_loader, test_loader



def train_test_set(mean=0., std=1, DATASET=MNIST, \
    do_target_transform=False, target_dim=28**2):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x : x.flatten() ) , 
        transforms.Lambda(lambda x : (x + mean)*std/ torch.sqrt (torch.mean(x**2) ) )
        ])
    #transforms.Normalize(mean, std)
    if do_target_transform :
        target_transform = transforms.Lambda(lambda l: math.sqrt(target_dim)*torch.eye(target_dim)[l])
    else:
        target_transform = transforms.Lambda(lambda x:x)

    train_set = DATASET(root='./data', 
                                            train=True,
                                            download=True,
                                            transform=transform,
                                            target_transform=target_transform)
    test_set = DATASET(root='./data', 
                                            train=False, 
                                            download=True, 
                                            transform=transform,
                                            target_transform=target_transform)

    return train_set, test_set

def train_test_loader(mean=0., std=1, batch_size=100, test_batch_size=500, num_workers=3, DATASET=MNIST, \
    do_target_transform=False, target_dim=28**2):
    train_set, test_set = train_test_set(mean, std, DATASET,\
    do_target_transform, target_dim,)
    train_loader = torch.utils.data.DataLoader(train_set,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_set, 
                                                batch_size=test_batch_size,
                                                shuffle=False, 
                                                num_workers=num_workers)

    return train_loader, test_loader



def train(loader, net, optimizer, criterion=nn.CrossEntropyLoss(), device="cuda", log_dir="log", max_iter=float("inf")):
    os.makedirs(log_dir, exist_ok=True)
    net.to(device)
    net.train()
    loss_file = "{}/train_loss.log".format(log_dir)
    touch(loss_file)
    mean_loss = 0.
    count = 0
    with open(loss_file, mode="a") as f:
        for x, l in tqdm(loader):
            if not count < max_iter: break
            x=x.to(device)
            l=l.to(device)
            optimizer.zero_grad()
            y = net(x)
            loss = criterion(y, l)
            loss.backward()
            optimizer.step()
            mean_loss += loss.item()
            count += 1
            f.write("{:.6f}\n".format(loss.item()))


    fig_file = "{}/train_loss.png".format(log_dir)
    plotter(loss_file,fig_file)

    mean_loss = mean_loss/count
    loss_file = "{}/mean_train_loss.log".format(log_dir)
    touch(loss_file)
    with open(loss_file, mode="a") as f:
        f.write("{:.6f}\n".format(mean_loss) )
    fig_file = "{}/mean_train_loss.png".format(log_dir)   
    plotter(loss_file,fig_file)
    
    
def test(loader, net, criterion=nn.CrossEntropyLoss(), device="cuda", log_dir="log",
debug=False): 
    os.makedirs(log_dir, exist_ok=True)
    net.eval()
    net.to(device)
    total = 0.
    correct = 0.
    loss = 0.
    accuracy_list = []
    print("start test ...")
    count = 0
    #import pdb; pdb.set_trace()
    if debug:
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
    with torch.no_grad():
        for x, l in tqdm(loader):
            x=x.to(device)
            l=l.to(device)
            y = net(x)
            total += l.shape[0]
            loss += criterion(y,l).item()
            _, predicted = torch.max(y.data, 1)
            #import pdb; pdb.set_trace()
            #if l.shape == y.shape:
            #    _, l = torch.where(l)
                #_, l = torch.max(l.data, 1)
            correct += (predicted == l).sum().item()    
            count += 1
            
            if debug:
                c = (predicted == l).squeeze()
                if len(l) == 1:
                    label = int(l)
                    class_correct[label] += c.item()
                    class_total[label] += 1
                else:
                    for i, label in enumerate(l):
                        class_correct[label] += c[i].item()
                        class_total[label] += 1


    accuracy = correct/total
    loss = loss/count

    print("test accuracy: {:.4f}".format(accuracy))
    acc_file = "{}/test_accuracy.log".format(log_dir)
    touch(acc_file)
    with open(acc_file, mode="a") as f:
        f.write("{:.4f}\n".format(accuracy))
    fig_file = "{}/test_accuracy.png".format(log_dir)
    plotter(acc_file,fig_file)

    print("test loss: {:.6f}".format(loss))
    loss_file = "{}/test_loss.log".format(log_dir)
    touch(loss_file)
    with open(loss_file, mode="a") as f:
        f.write("{:.6f}\n".format(loss))
    fig_file = "{}/test_loss.png".format(log_dir)
    plotter(loss_file,fig_file)


    if debug:
        for i in range(10):
            if class_total[i]  ==0 :
                print('class {} is not contained.'.format(i))
            else:
                acc_temp = class_correct[i] / class_total[i]
                print('Accuracy of class {} : {}'.format(i,acc_temp ))
                acc_file = "{}/test_accuracy_{}.log".format(log_dir,i)
                touch(acc_file)
                with open(acc_file, mode="a") as f:
                    f.write("{:.4f}\n".format(acc_temp))
                fig_file = "{}/test_accuracy_{}.png".format(log_dir,i)
                plotter(acc_file,fig_file)


    if debug:
        return accuracy, loss, class_correct, class_total
    else:
        return accuracy, loss



def register_save_var_grad(net, dirname):
    """
    Register a backward_hook  to layer.
    save variance of of grad_input[0].

    Pay attention to
    grad_output ==(backward)==>  grad_input

    """    
    filename = "{}/var_grad.log".format(dirname)
    #os.makedirs(filename, exist_ok=True)
    #touch(filename)
    def _save_var_grad(layer, grad_output, grad_input):
        var = torch.var(grad_input[0])
        with open(filename, mode="a") as f:
            f.write("{:.4e}\n".format(var.item()) ) 

    net.out.register_backward_hook(_save_var_grad)

def train_test(args,  net, dirname, dataset="FashionMNIST", use_MSE=True):
    print("------------------")
    print(dirname)
    device = args.device
    os.makedirs(dirname, exist_ok=True)

    if dataset == "FashionMNIST":
        DATASET = FashionMNIST
    elif dataset == "svhn" or "SVHN":
        DATASET = SVHN
    else:
        DATASET = dataset
    ### data and label 
    if use_MSE:
        ### transform label to vector for MSELoss
        train_loader, test_loader = train_test_loader(mean=0, std=1, batch_size=args.batch, DATASET=DATASET, \
            do_target_transform=True, target_dim=net.o_dim)
        #criterion = nn.MSELoss()        
        from src.loss.mseloss import HalfMSELoss 
        criterion = HalfMSELoss()
    else:
        train_loader, test_loader = train_test_loader(mean=0, std=1, batch_size=args.batch, DATASET=DATASET)
        criterion = nn.CrossEntropyLoss()

    print("start training ...")
    num_epochs = args.epoch
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, nesterov = args.nesterov)

    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, \
    #    600, eta_min=1e-4, last_epoch=-1)
    for i in range(num_epochs):
        print("epoch: {}, lr: {}".format(i, args.lr))
        train(train_loader, net, optimizer,criterion,\
            device=device, log_dir=dirname, max_iter=args.max_iter)        
        time.sleep(1)
        test_result =test(test_loader, net, criterion, device=device, log_dir=dirname)
        time.sleep(1)
        #scheduler.step()
    return test_result




def train_nets(args,
    only_plot_mean = False,
    use_MSE=False,
    o_dim = 10,
    ignore_last_layer=True):
    """
    Results are coollected at args.dirname.
    Check args.dirname/config.yml.
    """
    device = args.device
    dirname = get_plot_dir(args)
    additional_args= get_args_of_current_function()
    print(additional_args)
    write_config(dirname, additional_args)

    ### for plot_mean
    log_files = ["test_accuracy.log", "test_loss", "train_loss.log", "mean_train_loss.log"]
    
    if only_plot_mean:
        for log_file in log_files:
            plot_mean(root_dir=dirname, filename=log_file)
        return 
    net_name = eval(args.net)

    def _get_net():
        net = net_name(args.L, args.dim, o_dim=o_dim)
        return net

    if args.net == "MLP":
        net = _get_net()
        net.initialize_weight()
        net.to(device)
            
        def _run(net, dirname):
            test_acc, test_loss = train_test(args,  net, dirname,\
                use_MSE=use_MSE)        
        if args.num_running == 1:
            _run(net, dirname)
        else:
            for n in range(args.num_running):
                net = _get_net()
                net.to(device)
                net.initialize_weight()

                dirname_now = "{}/{}".format(dirname, n)
                _run(net, dirname_now)

            for log_file in log_files:
                plot_mean(root_dir=dirname, filename=log_file)
                test_acc, test_loss = train_test(args, net.i_var0, net, dirname,\
                    use_MSE=use_MSE)
            
    else:
        raise ValueError
