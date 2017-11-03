import sys
import argparse
import tensorflow as tf
from block_world import BlockWorldEnv as make
from dagger import Dagger
from dagger_policy import DaggerPolicy

def main(argv):
    env = make(run=False)

    dagger = Dagger(env,
                    DaggerPolicy,
                    num_rollouts = 25,
                    train_batch_size = 25,
                    train_epochs = 20,
                    iterations = 20,
                    train_report_frequency = 10,
                    dir_name = 'tmp_storage')

    dagger.learn_all_samples(save_file_name="dagger_block_world")

    env.close()

if __name__ == '__main__':
    main(sys.argv[1:])
