import sys
from args import get_args
import tensorflow as tf
from block_world import BlockWorldEnv as make
from dagger import Dagger
from dagger_policy import DaggerPolicy

def main(argv):
    args = get_args(argv)

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
                    **args)

    dagger.test("dagger_block_world")

    env.close()

if __name__ == '__main__':
    main(sys.argv[1:])