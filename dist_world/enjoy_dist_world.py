import gym
import sys
import argparse
import json

from block_world_core import env as make_block_env
from dist_world import DistworldEnv as make
from baselines import deepq


def main(argv):
    parser = argparse.ArgumentParser(description='block_world')

    parser.add_argument('--dims', type=int, default=3)
    parser.add_argument('--span', type=int, default=10)
    parser.add_argument('--single-dim-action', dest='single_dim_action', action='store_true')
    parser.add_argument('--no-single-dim-action', dest='single_dim_action', action='store_false')
    parser.add_argument('--episodes', type=int, default=100)

    parser.set_defaults(single_dim_action=False)
    cmd_args = parser.parse_args(argv)

    print(cmd_args)

    env = make(span=cmd_args.span,
               dims=cmd_args.dims,
               single_dim_action=cmd_args.single_dim_action)
    act = deepq.load("distworld_model.pkl")

    block_env = make_block_env()
    y_origin = 0.2
    block_env.set_params(tray_length=2. * float(cmd_args.span) * 0.1,
                         tray_width=2. * float(cmd_args.span) * 0.1,
                         tray_height=0.1,
                         rim_height=0.05,
                         rim_width=0.05)


    total_reward = 0
    for _ in range(cmd_args.episodes):
        obs, done = env.reset(), False
        j = block_env.json_objectinfo(0.05, env.finger_pos * 0.1, 0.03, env.target_pos * 0.1, [0, 2, -6], block_env.quaternion_id())
        block_env.reset_json_objectinfo(json.dumps(j))

        episode_rew = 0
        while not done:
            env.render()
            action = act(obs[None])[0]
            obs, rew, done, _ = env.step(action)

            episode_rew += rew
        print("Episode reward", episode_rew)
        total_reward += episode_rew

    print("average reward " + str(float(total_reward)/float(cmd_args.episodes)))

if __name__ == '__main__':
    main(sys.argv[1:])
