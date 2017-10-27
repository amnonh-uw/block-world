import sys
import argparse
import tensorflow as tf
from dist_world import DistworldEnv as make
from dagger import Dagger
from dagger_policy import DaggerPolicy

def main(argv):
    parser = argparse.ArgumentParser(description='block_world')
    parser.add_argument('--greedy', dest='greedy', action='store_true')
    parser.add_argument('--no-greedy', dest='greedy', action='store_false')
    parser.add_argument('--dims', type=int, default=3)
    parser.add_argument('--span', type=int, default=10)
    parser.add_argument('--max-timesteps', type=int, default=2000000)
    parser.add_argument('--exploration_fraction', type=float, default=0.1)
    parser.add_argument('--exploration_final_eps', type=float, default=0.01)
    parser.add_argument('--l2-penalty', type=float, default=None)
    parser.add_argument('--continous-actions', dest='continous_actions', action='store_true')
    parser.add_argument('--no-continous-actions', dest='continous_actions', action='store_false')
    parser.add_argument('--reach-minimum', type=float, default=0.1)
    parser.set_defaults(greedy=False)
    parser.set_defaults(continous_actions=False)
    cmd_args = parser.parse_args(argv)

    print(cmd_args)

    env = make(span=cmd_args.span,
               dims=cmd_args.dims,
               greedy=cmd_args.greedy,
               l2_penalty= cmd_args.l2_penalty,
               continous_actions=cmd_args.continous_actions,
               reach_minimum=cmd_args.reach_minimum)

    dagger = Dagger(env,
        num_rollouts = 25,
        train_batch_size = 25,
        train_iterations = 10000,
        iterations = 20)

    dagger.learn(DaggerPolicy, fname="dagger_dist_world")
    env.close()

if __name__ == '__main__':
    main(sys.argv[1:])
