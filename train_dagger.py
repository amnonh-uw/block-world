import sys
import argparse
import tensorflow as tf
from block_world import BlockWorldEnv as make
from dagger.dagger import Dagger
from dagger_policy import DaggerPolicy

def main(argv):
    parser = argparse.ArgumentParser(description='block_world')
    parser.add_argument('--show-obs', dest='show_obs', action='store_true')
    parser.add_argument('--no-show-obs', dest='show_obs', action='store_false')
    parser.add_argument('--run', dest='run', action='store_true')
    parser.add_argument('--no-run', dest='run', action='store_false')
    parser.add_argument('--reach-minimum', type=float, default=0.1)
    parser.set_defaults(show_obs=False)
    parser.set_defaults(run=True)
    args = parser.parse_args(argv)

    print(args)

    args.width = DaggerPolicy.width
    args.height = DaggerPolicy.height
    args.tray_length=3.0
    args.tray_width=2.0
    args.stereo_distance=0.5
    args.step_size = 0.1
    args.verbose = True

    env = make(**vars(args))

    dagger = Dagger(env,
                    DaggerPolicy,
                    num_rollouts = 25,
                    train_batch_size = 25,
                    train_epochs = 20,
                    iterations = 20,
                    train_report_frequency = 10,
                    dir_name = 'tmp_storage')

    dagger.learn(save_file_name="dagger_block_world")

    env.close()

if __name__ == '__main__':
    main(sys.argv[1:])
