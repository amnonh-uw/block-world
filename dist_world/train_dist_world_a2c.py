import sys
import argparse
from dist_world import DistworldEnv as make

from baselines.common import set_global_seeds
from baselines.a2c.a2c import learn
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from dist_world_a2c_policy import Policy as policy

def train(make_env, num_timesteps, seed, policy, lrschedule, num_cpu, vf_coef=0.5, ent_coef=0.01):
    def _make_env(rank):
        def _thunk():
            env = make_env()
            env.seed(seed + rank)
            return env
        return _thunk

    set_global_seeds(seed)
    env = SubprocVecEnv([_make_env(i) for i in range(num_cpu)])

    learn(policy, env, seed, nstack = 1, total_timesteps=num_timesteps, lrschedule=lrschedule, vf_coef=vf_coef, ent_coef=ent_coef)
    env.close()


def main(argv):
    parser = argparse.ArgumentParser(description='block_world')
    parser.add_argument('--greedy', dest='greedy', action='store_true')
    parser.add_argument('--no-greedy', dest='greedy', action='store_false')
    parser.add_argument('--dims', type=int, default=1)
    parser.add_argument('--span', type=int, default=10)
    parser.add_argument('--column-greedy', dest='column_greedy', action='store_true')
    parser.add_argument('--no-column-greedy', dest='column_greedy', action='store_false')
    parser.add_argument('--single-dim-action', dest='single_dim_action', action='store_true')
    parser.add_argument('--no-single-dim-action', dest='single_dim_action', action='store_false')
    parser.add_argument('--exploration_fraction', type=float, default=0.1)
    parser.add_argument('--exploration_final_eps', type=float, default=0.01)
    parser.add_argument('--l2-penalty', type=float, default=None)
    parser.add_argument('--continous-actions', dest='continous_actions', action='store_true')
    parser.add_argument('--no-continous-actions', dest='continous_actions', action='store_false')
    parser.add_argument('--no-stops', dest='no-stops', action='store_true')
    parser.add_argument('--no-no-stops', dest='no-stops', action='store_false')
    parser.add_argument('--reach-minimum', type=float, default=0.1)
    parser.add_argument('--max-far', type=float, default=None)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--lrschedule', help='Learning rate schedule', choices=['constant', 'linear'], default='constant')
    parser.add_argument('--million_frames', help='How many frames to train (/ 1e6)', type=int, default=10)
    parser.add_argument('--vf-coef', help='value function coefficient', type=float, default=0.5)
    parser.add_argument('--ent-coef', help='entropy  coefficient', type=float, default=0.01)
    parser.set_defaults(greedy=False)
    parser.set_defaults(column_greedy=False)
    parser.set_defaults(single_dim_action=False)
    parser.set_defaults(continous_actions=False)
    parser.set_defaults(no_stops=False)
    cmd_args = parser.parse_args(argv)

    print(cmd_args)

    def make_env():
        def _thunk():
            env = make(span=cmd_args.span,
                       dims=cmd_args.dims,
                       single_dim_action=cmd_args.single_dim_action,
                       greedy=cmd_args.greedy,
                       column_greedy=cmd_args.column_greedy,
                       l2_penalty=cmd_args.l2_penalty,
                       continous_actions=cmd_args.continous_actions,
                       reach_minimum=cmd_args.reach_minimum,
                       max_far=cmd_args.max_far)

            return env

        return _thunk

    train(make_env(),
          num_timesteps=int(1e6 * cmd_args.million_frames),
          seed=cmd_args.seed,
          policy=policy,
          lrschedule=cmd_args.lrschedule,
          num_cpu=16,
          vf_coef=cmd_args.vf_coef,
          ent_coef=cmd_args.ent_coef)

if __name__ == '__main__':
    main(sys.argv[1:])