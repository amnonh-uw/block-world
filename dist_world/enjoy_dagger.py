import sys
import argparse
import numpy as np
import tensorflow as tf

from block_world_core import env as make_block_env
from dist_world import DistworldEnv as make
from dagger import Dagger
from dagger_policy import policy

def main(argv):
    parser = argparse.ArgumentParser(description='block_world')

    parser.add_argument('--dims', type=int, default=3)
    parser.add_argument('--span', type=int, default=10)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--run', dest='run', action='store_true')
    parser.add_argument('--no-run', dest='run', action='store_false')

    cmd_args = parser.parse_args(argv)

    print(cmd_args)

    env = make(span=cmd_args.span,
               dims=cmd_args.dims)

    dagger = Dagger(env)

    with tf.Session() as sess:
        dagger.test(policy,"dagger_dist_world")

    # block_env = make_block_env(run=cmd_args.run)
    # y_origin = 0.2 + cmd_args.span * 0.1
    # block_env.set_params(tray_length=2. * float(cmd_args.span) * 0.1,
    #                      tray_width=2. * float(cmd_args.span) * 0.1,
    #                      tray_height=0.1,
    #                      rim_height=0.05,
    #                      rim_width=0.05)
    #
    #
        total_reward = 0
    #
    # def shift_y(pos):
    #     shifted = np.copy(pos)
    #     shifted[1] += y_origin
    #     return shifted

        for _ in range(cmd_args.episodes):
            obs, done = env.reset(), False
            # block_env.clear_tray()
            # pos = env.finger_pos
            # pos[1] += y_origin
            # block_env.set_finger(shift_y(env.finger_pos.astype(float) * 0.1))
            # block_env.set_target(shift_y(env.target_pos.astype(float) * 0.1))

            episode_rew = 0
            while not done:
                env.render()
                action = dagger.eval_policy(obs)[0]
                obs, rew, done, _ = env.step(action)
                episode_rew += rew

                # action = env.map_discrete_action(action)
                # block_env.move_finger(action.astype(float) * 0.1)

            print("Episode reward", episode_rew)
            total_reward += episode_rew

    print("average reward " + str(float(total_reward)/float(cmd_args.episodes)))
    env.close()
    # block_env.close()

if __name__ == '__main__':
    main(sys.argv[1:])
