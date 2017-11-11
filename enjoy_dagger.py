import sys
from args import get_args, env_args
import tensorflow as tf
from block_world import BlockWorldEnv as make
from dagger import Dagger
from dagger_policy import DaggerPolicy

def main(argv):
    args = get_args(argv)

    x = env_args()
    print(x)
    env = make(**vars(env_args()))

    dagger = Dagger(env,
                    DaggerPolicy,
                    **vars(args))

    dagger.test("dagger_block_world")

    env.close()

if __name__ == '__main__':
    main(sys.argv[1:])
