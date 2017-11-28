import sys
from args import get_args, env_args
from block_world import BlockWorldEnv as make
from dagger import Dagger
from importlib import import_module

def main(argv):
    args = get_args(argv)

    policy_mod = import_module(args.policy_source)
    policy_class = getattr(policy_mod, 'DaggerPolicy')
    policy = policy_class(**vars(args))

    x = env_args(args)
    env = make(**vars(x))

    args.iterations = 0
    print(args.num_rollouts)

    dagger = Dagger(env,
                    policy,
                    **vars(args))

    dagger.learn()

    env.close()

if __name__ == '__main__':
    main(sys.argv[1:])
