###################################
##### gather results and plot #####
###################################
import pandas
import yaml
import shutil
import os
import glob


def gather_result(job_name):
    """
    For job_name, return path to the directories
    which contain the job-result
    """
    dirnames = glob.glob("plot/*")
    out_dirnames = []
    for dirname in dirnames:
        config_file = "{}/config.yml".format(dirname)
        try:
            with open(config_file) as f:
                try :
                    obj = yaml.load(f)

                except Exception as e:
                    print("Error of yaml.load. Check config file:{}".format(config_file))

                if obj["args"].job_name == job_name:
                    out_dirnames.append(dirname)

                    

        except Exception as e:        
            0

    return out_dirnames



def job_to_pkl(job_names):
    pkl_name = ""
    for job_name in job_names:
        pkl_name = "{}_{}".format(pkl_name, job_name)
    pkl_name = pkl_name + ".pkl"
    os.makedirs("result/", exist_ok=True)
    return "result/{}".format(pkl_name)


def convert_results_to_pkl(args, index, job_names, log_files, only_gather_out_of_ntk=False,N=1, thres=10):
    ### set up jobname
    ### we train nets for last job_name
    ### for plot

    ### index.append("max_eig")
    dirnames_list = []
    for job_name in job_names:
        print("job_name:", job_name)
        dirnames = gather_result(job_name)
        dirnames_list.append(dirnames)


    df = pandas.DataFrame()
    for dirnames in dirnames_list:
        df = _append_results_to_pandas(df, dirnames, log_files, index,\
            thres=thres)

    pkl_name = job_to_pkl(job_names)
    df.to_pickle(pkl_name)



def _append_results_to_pandas(df, dirnames, log_files, index, thres):
    """
    Parameters:
        df :  pandas.DataFrame            
    """
    assert len(log_files) == 3  ### to be removed


    if "max_eig" in index:
        max_eig_file = "plot/ell_vs_max_eig/20200319172817/max_eigs.log"
        with open (max_eig_file) as f:
            a = f.readlines()    
        max_eigs = [float(x) for x in a]


    for dirname in dirnames:
        config = "{}/config.yml".format(dirname)
        with open(config, "r") as f:
            obj = yaml.load(f)
        ob_args =  obj["args"]

        results = []
        try:

            for log_file in log_files:
                with open("{}/{}".format(dirname, log_file)) as f:
                    a = f.readlines()
                vals = [float(x) for x in a]
                value = vals[0]
                value = min(thres, value)
                results.append(value)

            data = [ob_args.job_name, ob_args.L, ob_args.lr,\
                results[0], results[1], results[2]]
            if "max_eig" in index:
                data.append(max_eigs[int(ob_args.L) -1 ] ) ### [ToDo]  Conflict ? 
            if "batch" in index:
                data.append(int(ob_args.batch) )
            assert len(data) == len(index)
            s = pandas.Series(data, index = index, name=dirname)
            df = df.append(s)
        except FileNotFoundError:
            print("FileNotFoundError of logfile at dirname=", dirname)        
    return df



