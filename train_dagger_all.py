import sys
import imp
from args import get_args
from block_world import BlockWorldEnv as make
from dagger import Dagger
from dagger_policy_conv import DaggerPolicy

def main(argv):
    args = get_args(argv)

    print(args)
    policy = imp.load_source("DaggerPolicy", args.policy_source)

    dagger = Dagger(None,
                    policy.DaggerPolicy,
                   **vars(args))

    # dagger.learn_all_samples(save_file_name="dagger_block_world", load_file_name="dagger_block_world")
    dagger.learn_all_samples()

if __name__ == '__main__':
    main(sys.argv[1:])
