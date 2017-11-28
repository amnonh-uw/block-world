import sys
import argparse

def get_args(argv):
    parser = argparse.ArgumentParser(description='block_world')
    parser.add_argument('--show-obs', dest='show_obs', action='store_true')
    parser.add_argument('--no-show-obs', dest='show_obs', action='store_false')
    parser.add_argument('--run', dest='run', action='store_true')
    parser.add_argument('--no-run', dest='run', action='store_false')
    parser.add_argument('--reach-minimum', type=float, default=0.1)
    parser.add_argument('--num-rollouts', type=int, default=25)
    parser.add_argument('--batch-size', type=int, default=25)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--learning-rate', type=float, default=1.0e-5)
    parser.add_argument('--iterations', type=int, default = 20)
    parser.add_argument('--report_frequency', type=int, default=10)
    parser.add_argument('--dir-name', type=str, default=None)
    parser.add_argument('--max-samples', type=str, default=None)
    parser.add_argument('--policy-source', type=str, default='policies.null')
    parser.add_argument('--save-file-name', type=str, default=None)
    parser.add_argument('--width', type=int, default = 224)
    parser.add_argument('--height', type=int, default = 224)
    parser.add_argument('--max-objects', type=int, default = 5)
    parser.set_defaults(show_obs=False)
    parser.set_defaults(run=True)

    args =  parser.parse_args(argv)

    if args.save_file_name is None:
        args.save_file_name = args.policy_source + "_save"

    if args.dir_name is None:
        args.dir_name = "storage{}x{}".format(args.width, args.height)

    return args

def env_args(args):
    env_args = argparse.Namespace()
    
    env_args.tray_length=3.0
    env_args.tray_width=2.0
    env_args.stereo_distance=0.5
    env_args.step_size = 0.1
    env_args.reach_minimum = 0.1
    env_args.verbose = False
    env_args.run = True
    env_args.width = args.width
    env_args.height = args.height
    env_args.max_objects = args.max_objects

    return env_args
