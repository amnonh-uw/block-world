import gym
import sys
import argparse
import tensorflow as tf
import tensorflow.contrib.layers as layers
from baselines import deepq
from dist_world import DistworldEnv as make

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

    model = deepq.models.mlp(hiddens=[64, 64, 64, 64])

    act = deepq.learn(
        env,
        q_func=model,
        lr=1e-4,
        max_timesteps=cmd_args.max_timesteps,
        buffer_size=10000,
        exploration_fraction=cmd_args.exploration_fraction,
        exploration_final_eps=cmd_args.exploration_final_eps,
        train_freq=4,
        learning_starts=10000,
        target_network_update_freq=1000,
        gamma=0.99,
        prioritized_replay=True
    )
    act.save("distworld_model.pkl")
    env.close()

if __name__ == '__main__':
    main(sys.argv[1:])
