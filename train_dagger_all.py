import sys
from args import get_args
from block_world import BlockWorldEnv as make
from dagger import Dagger
from dagger_policy_nets import DaggerPolicy

def main(argv):
    env = make(run=False)
    
    args = get_args(argv)

    print(args)

    dagger = Dagger(env,
                    DaggerPolicy,
                   **vars(args))

    # dagger.learn_all_samples(save_file_name="dagger_block_world", load_file_name="dagger_block_world")
    dagger.learn_all_samples(save_file_name="dagger_block_world")

    env.close()

if __name__ == '__main__':
    main(sys.argv[1:])
