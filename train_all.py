import sys
from importlib import import_module
from args import get_args
from dagger import Dagger

def main(argv):
    args = get_args(argv)

    print(args)
    policy_mod = import_module(args.policy_source)
    policy = getattr(policy_mod, 'DaggerPolicy')

    dagger = Dagger(None,
                    policy,
                   **vars(args))

    # dagger.learn_all_samples(save_file_name="dagger_block_world", load_file_name="dagger_block_world")
    dagger.learn_all_samples()

if __name__ == '__main__':
    main(sys.argv[1:])
