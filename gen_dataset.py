import argparse
import pickle
import sys
import numpy as np

from block_world_core import env as make

def dataset_files(dataset):
    idx_file = dataset + ".idx"
    data_file = dataset + ".data"

    return data_file, idx_file

class Datasetinfo:
    def __init__(self, episodes):
        self.epsisodes = episodes

def gen_dataset(argv):
    parser = argparse.ArgumentParser(description='gen_dataset')
    parser.add_argument('--pos-unit', type=float, default=0.1)
    parser.add_argument('--rot-unit', type=float, default=1)
    parser.add_argument('--no-units', dest='no_units', action='store_true')
    parser.add_argument('--run', dest='run', action='store_true')
    parser.add_argument('--no-run', dest='run', action='store_false')
    parser.add_argument('--dataset', default="dataset", type=str)
    parser.add_argument('--episodes', default=100, type=int)
    parser.set_defaults(no_units=False)
    parser.set_defaults(run=True)
    cmd_args = parser.parse_args(argv)

    print("generating {} samples to {}".format(cmd_args.episodes, cmd_args.dataset))
    data_file, idx_file  = dataset_files(cmd_args.dataset)
    index = np.zeros([cmd_args.episodes], dtype=int)

    env = make(run=cmd_args.run)

    with open(data_file, 'wb') as data:
        for i in range(0, cmd_args.episodes):
            env.reset()
            info = env.get_json_objectinfo()
            index[i] = data.tell()
            pickle.dump(info, data, pickle.HIGHEST_PROTOCOL)

    dataset_info = Datasetinfo(cmd_args.episodes)
    with open(idx_file, 'wb') as idx:
        pickle.dump(index, idx, pickle.HIGHEST_PROTOCOL)
        pickle.dump(dataset_info, idx, pickle.HIGHEST_PROTOCOL)

    env.close()
        
if __name__ == "__main__":
    gen_dataset(sys.argv[1:])
