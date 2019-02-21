import os
import glob
import json
import torch

def hdfs_save(hdfs_dir, filename):
    os.system('/opt/hadoop/latest/bin/hdfs dfs -copyFromLocal -f {} {}'.format(filename, hdfs_dir))

def hdfs_load(hdfs_dir, local_dir):
    hdfs_expt = '{}/{}'.format(hdfs_dir, local_dir)
    os.system('/opt/hadoop/latest/bin/hdfs dfs -copyToLocal {} .'.format(hdfs_expt))


class Logger:
    def __init__(self, outdir):
        self.outdir = outdir
        self.save_folder = '{0}/saves/'.format(self.outdir)
        self.log_file = '{0}/log.txt'.format(self.outdir)

        self.hdfs_found = True
        if not os.path.exists(self.save_folder):
            self.hdfs_found = False
            os.makedirs(self.save_folder)

        self.ldict = {}

    def save_config(self, args):
        with open(os.path.join(self.save_folder, 'config.json'), 'w') as f:
            config = {k: v for (k, v) in vars(args).items() if k != 'device'}
            config.update(device=args.device.type)
            json.dump(config, f, indent=2)

    def load_checkpoint(self):
        pfiles = [x for x in glob.glob(self.save_folder + "*.pt") if 'final' not in x]
        fname = max(pfiles, key=lambda x: int(x.split('-')[-1].split('.')[0]))
        batch = int(fname.split('-')[-1].split('.')[0])

        # updating log file
        with open(self.log_file, 'rb') as f:
            lines = f.readlines()
        with open(self.log_file, 'wb') as f:
            for line in lines[:batch]:
                f.write(line)
        
        # updating ldict
        lines = [x.rstrip() for x in open(self.log_file).readlines()]
        for l in lines:
            sline = l.split()
            keys = [sline[i] for i in range(0, len(sline), 2)]
            values = [float(sline[i]) for i in range(1, len(sline), 2)]
            for k,v in zip(keys, values):
                if k not in self.ldict:
                    self.ldict[k] = []
                self.ldict[k].append(v)

        return fname, batch

    def save_policy(self, batch, policy):
        with open(os.path.join(self.save_folder, 'policy-{0}.pt'.format(batch)), 'wb') as f:
            torch.save(policy.state_dict(), f)

    def set_keys(self, keys):
        for k in keys:
            self.ldict[k] = []

    def logkv(self, key, value):
        self.ldict[key].append(value)

    def print_results(self):
        print_string = ''
        for k in self.ldict:
            print_string += '{} {} '.format(k, self.ldict[k][-1])
        print(print_string)
        with open(self.log_file, 'a') as f:
            print(print_string, file=f)