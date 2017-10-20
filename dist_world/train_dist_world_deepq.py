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
    parser.add_argument('--column-greedy', dest='column_greedy', action='store_true')
    parser.add_argument('--no-column-greedy', dest='column_greedy', action='store_false')
    parser.add_argument('--single-dim-action', dest='single_dim_action', action='store_true')
    parser.add_argument('--no-single-dim-action', dest='single_dim_action', action='store_false')
    parser.add_argument('--max-timesteps', type=int, default=2000000)
    parser.add_argument('--exploration_fraction', type=float, default=0.1)
    parser.add_argument('--exploration_final_eps', type=float, default=0.01)
    parser.add_argument('--l2-penalty', type=float, default=None)
    parser.add_argument('--continous-actions', dest='continous_actions', action='store_true')
    parser.add_argument('--no-continous-actions', dest='continous_actions', action='store_false')
    parser.add_argument('--no-stops', dest='no-stops', action='store_true')
    parser.add_argument('--no-no-stops', dest='no-stops', action='store_false')
    parser.add_argument('--reach-minimum', type=float, default=0.1)
    parser.add_argument('--max-far', type=float, default=None)
    parser.set_defaults(greedy=False)
    parser.set_defaults(column_greedy=False)
    parser.set_defaults(single_dim_action=False)
    parser.set_defaults(continous_actions=False)
    parser.set_defaults(no_stops=False)
    cmd_args = parser.parse_args(argv)

    print(cmd_args)

    env = make(span=cmd_args.span,
               dims=cmd_args.dims,
               single_dim_action=cmd_args.single_dim_action,
               greedy=cmd_args.greedy,
               column_greedy=cmd_args.column_greedy,
               l2_penalty= cmd_args.l2_penalty,
               continous_actions=cmd_args.continous_actions,
               reach_minimum=cmd_args.reach_minimum,
               max_far=cmd_args.max_far)

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
