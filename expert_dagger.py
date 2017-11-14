import sys
from args import get_args, env_args
from block_world import BlockWorldEnv as make
from dagger import Dagger
from importlib import import_module

def main(argv):
    args = get_args(argv)

    x = env_args()
    x.max_objects = 0
    print(x)
    env = make(**vars(x))

    policy_mod = import_module(args.policy_source)
    policy = getattr(policy_mod, 'DaggerPolicy')

    args.iterations = 0
    print(args.num_rollouts)

    dagger = Dagger(env,
                    policy,
                    **vars(args))

    dagger.learn()

    env.close()

if __name__ == '__main__':
    main(sys.argv[1:])
