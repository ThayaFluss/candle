import os

def touch(path):
    if os.path.isfile(path):
        pass
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="UTF-8") as f:
            pass




import yaml
import time, datetime



def write_config(dirname, additional_args):
    os.makedirs(dirname, exist_ok=True)
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





