import sys
import argparse

def get_args(argv):
    parser = argparse.ArgumentParser(description='block_world')
    parser.add_argument('--show-obs', dest='show_obs', action='store_true')
    parser.add_argument('--no-show-obs', dest='show_obs', action='store_false')
    parser.add_argument('--run', dest='run', action='store_true')
    parser.add_argument('--no-run', dest='run', action='store_false')
    parser.add_argument('--reach-minimum', type=float, default=0.1)
    parser.add_argument('--num-rolloutsm', type=int, default=25)
    parser.add_argument('--batch-size', type=int, default=25)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--iterations', type=int, default = 20)
    parser.add_argument('--report_frequency', type=int, default=10)
    parser.add_argument('--dir-name', type=str, default='tmp_storage')
    parser.add_argument('--max-samples', type=str, default=None)
    parser.add_argument('--policy-source', type=str, default='dagger_policy')
    parser.add_argument('--save-file-name', type=str, default=None)
    parser.set_defaults(show_obs=False)
    parser.set_defaults(run=True)

    args =  parser.parse_args(argv)

    if args.save_file_name is None:
        args.save_file_name = args.policy_source + "_save"

    return args

def env_args():
    args = argparse.Namespace()
    
    args.tray_length=3.0
    args.tray_width=2.0
    args.stereo_distance=0.5
    args.step_size = 0.1
    args.reach_minimum = 0.1
    args.verbose = True
    args.run = True

    return args
