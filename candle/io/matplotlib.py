import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import glob
from .util import touch

def plotter(log_file, fig_file=None,
    xlabel=None, ylabel=None):
    """
    A simple util to load data and write it to fig_file.
    """
    with open(log_file) as f:
        a = f.readlines()
    result = [ float(x) for x in a ]
    fig = plt.figure()
    plt.plot(result)
    if not xlabel is None: plt.xlabel(xlabel)
    if not ylabel is None: plt.ylabel(ylabel)
    if fig_file is None:
        print("(plotter)fig_file is None. Set fig_file = {}.png".format(log_file))
        fig_file = "{}.png".format(log_file)
    print("(plotter){} ---> {}".format(log_file, fig_file))
    plt.savefig(fig_file)
    fig.clf()
    plt.close()
    del fig

def multi_plotter(log_files,fig_file, x=None):
    results = []
    plt.figure()

    for idx, log_file in  enumerate(log_files):
        with open(log_file) as f:
            a = f.readlines()
            result = [float(x) for x in a] 
            #print(result.__len__())
            if x is None:
                plt.plot(result, label="{}".format(log_file))
            else:
                plt.plot(x,result, label="{}".format(log_file))
    plt.legend()
    plt.savefig(fig_file)
    plt.close()


def plot_mean(root_dir, filename, max_len=None):
    results = []
    log_files = glob.glob("{}/*/{}".format(root_dir, filename))
    if len(log_files) == 0:
        print("(plot_mean)No such logfiles:{}".format(filename))
        return

    plt.figure()
    for idx, log_file in  enumerate(log_files):
        with open(log_file, "r") as f:
            a = f.readlines()
            result = [float(x) for x in a] 
            if max_len is not None and result.__len__() > max_len:
                print("(plot_mean)[WARNING]len(reuslt) in {}: {}".format(log_file, result.__len__()) ) 
                continue
            results.append(result)

    mean = np.array(results).mean(0)
    out_log = "{}/{}".format(root_dir, filename)
    touch(out_log)
    with open(out_log, mode="a") as f:
        for m in mean:
            f.write("{:.6f}\n".format(m))


    b, ext = os.path.splitext(out_log)
    out_fig = "{}.png".format(b)
    plotter(out_log, out_fig)


def file_to_array(log_file):
    with open(log_file) as f:
        a = f.readlines()
    result = [ float(x) for x in a ]
    result = np.array(result)
    return result


