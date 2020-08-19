import os, sys
sys.path.append( os.path.dirname(__file__) + "/../" )

import torch
import argparse
import yaml
import datetime, time
import inspect


def parser_init():
    """
    e.g.
        parser = setup_parser(args)
        args = parser.parse_args()
    or 
        parser = parser_init()
        args = parser.parse_args(sys.argv[1:])
    """
    parser = argparse.ArgumentParser(description='Process args for ML-experiments.')
    ###  often used
    parser.add_argument('--job_name',  type=str, default="temp",
                        help='job name to identify experiments (default: %(default)s)')
    parser.add_argument('--device_id',  type=int, default=0,
                        help='GPU id. Set -1 for cpu (default: %(default)s)')

    parser.add_argument('--batch',  type=int, default=500,
                        help='batch size for training (default: %(default)s)')
    parser.add_argument('--lr',  type=float, default=1e-4,
                        help='learning rate for training (default: %(default)s)')
    parser.add_argument('--epoch',  type=int, default=5,
                        help='max epoch (default: %(default)s)')


    ### possilbly unused
    parser.add_argument('--max_iter',  type=float, default=float("inf"),
                        help=' max number of iterations per epoch (default: %(default)s)')
    parser.add_argument('--num_running',  type=int, default=1,
                        help='number of running experiments  (default: %(default)s)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help=' momentum of SGD (default: %(default)s)')
    parser.add_argument('--nesterov', action='store_true', 
                        help=' use this to use Nestreov option of SGD')


    ### not general
    parser.add_argument('--net',  type=str, default="MLP",
                        help='Class-name of Network (default: %(default)s)')

    parser.add_argument('--L', metavar='N', type=int, default=100,
                        help='Number of Layers of Network (default: %(default)s)')
    parser.add_argument('--dim', metavar='N', type=int, default=784,
                        help='Width of Input (default: %(default)s)')



    return parser


def parse_device(args):
    if args.device_id >= 0:
        assert torch.cuda.is_available()
        device = "cuda:{}".format(args.device_id)
    else:
        device =  "cpu"
    return device


def argsp():
    parser = parser_init()
    args = parser.arg_parse()
    device = parse_device(args)
    args.device = device
    return args


def get_args_of_current_function():
    current_frame = inspect.currentframe()
    parent_frame = current_frame.f_back
    info = inspect.getargvalues(parent_frame)
    return {key: info.locals[key] for key in info.args}


def write_config(dirname):
    os.makedirs(dirname, exist_ok=True)
    additional_args = get_args_of_current_function()
    with open("{}/config.yml".format(dirname), "w" ) as f:
        yaml.dump(additional_args,f)
    return


def get_plot_dir(args):
    """
    TODO:Duplicated
    """
    today = datetime.datetime.fromtimestamp(time.time())
    t = today.strftime('%Y%m%d%H%M%S')
    dirname = "plot/{}_intvl-{}_L-{}_{}".format(args.net, args.interval, args.L,t)
    return dirname


