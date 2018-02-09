#!/usr/bin/env python

import sys
import platform
import subprocess
import requests
import argparse
import io
from PIL import Image
from time import sleep
import numpy as np
import json
from a_star.a_star_bw import find_next_step

class env:
    last_port = 9000

    def __init__(self, log="/dev/stdout", pos_unit=None, rot_unit=None,run = True, verbose=False, params_args=None):
        self.port = env.last_port
        self.uri = "http://localhost:" + str(self.port) + "/"
        env.last_port += 1
        self.pos_unit = pos_unit
        self.rot_unit = rot_unit
        self.log = log
        self.verbose = verbose

        self.first_connection = True
        if run:
            self.run()

        if params_args != None:
            self.set_params(**params_args)

        self.obs = dict()

    def run(self):
        if self.is_running():
            print("run - its already running, do not run again")
            return

        from inspect import getfile
        import os.path
        module_path = getfile(sys.modules[__name__])
        exe = os.path.dirname(module_path)
        if exe == "":
            exe = "."

        if platform.system() == 'Darwin':
            exe += "/builds/osx_player.app/Contents/MacOS/osx_player"
        else:
            exe += "/builds/linux_player.x86_64"
        logopt = "-logFile "
        args = "-screen-width 640 -screen-height 480 -screen-quality ultra -screen-fullscreen 0"
        shell_command = "export PORT=" + str(self.port) + " && " + exe + " "
        if self.log:
            shell_command += logopt + " " + self.log + " " + args
        else:
            shell_command += " " + args

        self.process = subprocess.Popen(shell_command, shell=True)

    def is_running(self):
        try:
            r = requests.get(self.uri, timeout=1)
            return True

        except requests.exceptions.ConnectionError:
            return False

        except requests.exceptions.Timeout:
            return False

    def do(self, command, args=None):
        not_done = True
        count = 0

        args = self.make_string(args)
        if self.verbose:
            print("{} {}".format(command,"" if args is None else args))


        while not_done:
            try:
                r = requests.post(self.uri, data={"command": command, "args": args}, timeout=60)
                not_done = False
                self.first_connection = False
            except requests.exceptions.ConnectionError:
                count += 1
                if count >= 50 or not self.first_connection:
                    print("ConnectionError uri=" + str(self.uri))
                    return None
                sleep(0.1)
            except requests.exceptions.Timeout:
                print("Timeout on request!!! - retry")

        if r.status_code != 200:
            print('Status:', r.status_code, 'Problem with the request. Exiting.')
            return None

        return r

    def make_string(self, args):
        if args is None:
            return args

        if type(args) is str:
            return args

        if type(args) is np.ndarray:
            args_arrstr = np.char.mod('%f', args)
            # combine to a string
            return ",".join(args_arrstr)

        if type(args) is float:
            return(str(args))

        print("unknwon data type {} in make_string".format(type(args)))
        return None

    def round_unit(self, v, unit):
        if unit is None:
            return v

        if v is None:
            return None

        v = float(v)
        d = v // unit
        r = v % unit

        if r >= unit / 2.:
            d += 1

        return unit * d

    def round_pose(self, pose):
        pose = self.make_string(pose)
        i = 0
        s = ""
        unit = self.pos_unit
        for v in self.clean_split(pose, ","):
            s += str(self.round_unit(v, unit))
            s += ","
            i += 1
            if i == 3:
                unit = self.rot_unit

        return s[:-1]

    def quit(self):
        self.do("quit")

    def close(self):
        self.do("quit")

    def set_params(self, **kwargs):
        for key in kwargs:
            print("{}={}".format(key, kwargs[key]))
            self.do(key, str(kwargs[key]))

    def reset(self, **kwargs):
        self.set_params(**kwargs)
        r = self.do("reset")
        self.extract_response(r.json())

    def clear_tray(self):
        r = self.do("clear_tray")
        self.extract_response(r.json())

    def rotate_joint(self, args):
        r = self.do("rotate_joint", args)
        self.extract_response(r.json())

    def check_occupied(self, inc_pose):
        r = self.do("check_occupied", self.make_string(inc_pose))
        return self.extract_bool(r.json(), "b")

    def move_cams(self, inc_pose):
        inc_pose = self.round_pose(inc_pose)
        r = self.do("move_cams", str(inc_pose))
        self.extract_response(r.json())

    def res(self, res):
        p = res.split(",", 2)
        self.do("width", p[0])
        self.do("height", p[1])

    def get_json_params(self, args):
        r = self.do("get_json_params")
        return r.json()

    def get_json_objectinfo(self):
        r = self.do("get_json_objectinfo")
        return r.json()

    def extract_cam(self, data, key):
        if key in data:
            cam_stream = io.BytesIO(bytearray(data[key]))
            return Image.open(cam_stream)
        else:
            print("can't find cam {}".format(key))
            return None

    def extract_float(self, data, key):
        if key in data:
            return float(data[key])
        else:
            return None

    def extract_string(self, data, key):
        if key in data:
            return str(data[key])
        else:
            return None

    def extract_vector3(self, data, key):
        if key in data:
            data = data[key]
            x = self.extract_float(data, "x")
            y = self.extract_float(data, "y")
            z = self.extract_float(data, "z")
            return np.array([x, y, z])
        else:
            return None

    def extract_bool(self, data, key):
        return(data[key])

    def save_cams(self, path):
        path = path.replace(".tfrecord", '')
        self.leftcam.save(path + 'left.png')
        self.rightcam.save(path + 'right.png')
        self.centercam.save(path + 'center.png')
        self.depthcam.save(path + 'depth.png')
        self.raw_multdepthcam.save(path + 'rawdepth.png')
        self.multichanneldepthcam.save(path + 'multdepth.tiff')
        self.normalcam.save(path + 'normals.png')
        pass

    def save_positions(self, path):
        path = path.replace(".tfrecord", '')
        with open(path + 'positions.txt', 'w') as f:
            f.write("finger screen {} target screen {}".format(self.finger_screen_pos, self.target_screen_pos))

    def decode_twochanneldepth(self, img):
        source = np.asarray(img, dtype=np.uint32)
        # red channel is lowBits
        # green channel is highBits
        # alpha channel is 255 for valid values (i.e. rendered objects vs infinity) 0 otherwise

        depth = source[:,:,0] + 256 * source[:,:,1]
        depth[source[:,:,3] != 255] = 256*256 - 1
        depth = depth.astype(np.uint16)
        depth_img = Image.fromarray(depth)

        return depth_img

    def extract_response(self, data):
        self.obs['leftcam'] = self.leftcam = self.extract_cam(data, "leftcam")
        self.obs['rightcam'] = self.rightcam = self.extract_cam(data, "rightcam")
        self.obs['centercam'] = self.centercam = self.extract_cam(data, "centercam")
        self.depthcam = self.extract_cam(data, "depthcam")
        self.raw_multdepthcam = self.extract_cam(data, "multichanneldepthcam")
        self.obs['multichanneldepthcam'] = self.multichanneldepthcam = self.decode_twochanneldepth(self.raw_multdepthcam)
        self.obs['normalcam'] = self.normalcam = self.extract_cam(data, "normalcam");
        self.obs['no_collision'] =  self.no_collision = self.extract_bool(data, "no_collision")
        self.collision = not self.no_collision
        self.obs['object_collision'] = self.object_collision = self.extract_bool(data, "object_collision")
        self.obs['collision_name'] = self.collision_name = self.extract_string(data, "collision_name")
        self.obs['collision_point'] = self.collision_point = self.extract_vector3(data, "collision_point")
        self.obs['collision_distance'] = self.collision_distance = self.extract_float(data, "collision_distance")
        self.obs['collision_screen_point'] = self.collision_screen_point = self.extract_vector3(data, "collision_screen_point")


    def obs_dict(self):
        return self.obs

    def clean_split(self, s, delim):
        s = s.replace(" ", "")
        s = s.replace("(", "")
        s = s.replace(")", "")

        return s.split(delim)


import pygame
import xbox360_controller
def joystick(x, d):
    joints = [ "right_s0", "right_s1", "right_e0", "right_e1", "right_w0", "right_w1", "right_w2"] 
    joint = 0
    
    pygame.init()
    my_controller = xbox360_controller.Controller(0)
    done = False
    while not done:
        # event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                 done=True

            if event.type == pygame.JOYBUTTONDOWN:
                if event.button == xbox360_controller.A:
                    joint += 1
                if event.button == xbox360_controller.B:
                    joint -= 1
                if event.button == xbox360_controller.Y:
                    done = True

                if joint < 0:
                    joint += len(joints)
                if joint >= len(joints):
                    joint -= len(joints)

                print("joint is {}", joints[joint]);


        # handle joysticks
        left_x, left_y = my_controller.get_left_stick()
        if left_x != 0:
            s = joints[joint] + "," + str(left_x)
            print(s)
            x.rotate_joint(s)
            d.show(x)

def env_test(argv):
    parser = argparse.ArgumentParser(description='block_world')
    parser.add_argument('--reset', dest='reset', action='store_true')
    parser.add_argument('--no-reset', dest='reset', action='store_false')
    parser.add_argument('--pos-unit', type=float, default=0.1)
    parser.add_argument('--rot-unit', type=float, default=1)
    parser.add_argument('--no-units', dest='no_units', action='store_true')
    parser.add_argument('--show-obs', dest='show_obs', action='store_true')
    parser.add_argument('--no-show-obs', dest='show_obs', action='store_false')
    parser.add_argument('--run', dest='run', action='store_true')
    parser.add_argument('--no-run', dest='run', action='store_false')
    parser.add_argument('--verbose', dest='verbose', action='store_true')
    parser.add_argument('--no-verbose', dest='verbose', action='store_false')
    parser.set_defaults(reset=True)
    parser.set_defaults(no_units=False)
    parser.set_defaults(show_obs=False)
    parser.set_defaults(run=True)
    parser.set_defaults(verbose=True)
    cmd_args = parser.parse_args(argv)


    if cmd_args.no_units:
        cmd_args.pos_unit = None
        cmd_args.rot_unit = None

    x = env(pos_unit=cmd_args.pos_unit, rot_unit=cmd_args.rot_unit, run=cmd_args.run)
    d = _display(show_obs=cmd_args.show_obs)

    if cmd_args.reset:
        x.reset(tray_length=1.50, tray_width=1.0, stereo_distance=0.5, width=224, height=224)
        d.show(x)

    var_dict = dict()

    from cmd import Cmd
    import random

    class MyPrompt(Cmd):
        def do_quit(self, args):
            x.quit()
            raise SystemExit
    
        def do_joystick(self, args):
            joystick(x, d)

        def do_EOF(self, args):
            x.quit()
            print("")
            raise SystemExit

        def do_reset(self, args):
            if args:
                x.reset(max_objects=int(args))
            else:
                x.reset(tray_length=3.0, tray_width=2.0)
            d.show(x)

        def do_get_json_params(self, args):
            json_value = x.get_json_params(args)
            if args:
                var_dict[args] = json_value
            print(json.dumps(json_value))

        def do_get_json_objectinfo(self, args):
            json_value = x.get_json_objectinfo()
            if args:
                var_dict[args] = json_value
            print(json.dumps(json_value))

        def do_clear_tray(self):
            x.clear_tray()
            d.show(x)

        def do_rotate_joint(self, args):
            x.rotate_joint(args)
            d.show(x)

        def do_move_cams(self, args):
            x.move_cams(args)
            d.show(x)

        def do_res(self, args):
            x.res(args)

        def do_up(self, args):
            x.move_finger("0,0.1,0")

        def do_down(self, args):
            x.move_finger("0,-0.1,0")

        def do_left(self, args):
            x.move_finger("-0.1,0,0")

        def do_right(self, args):
            x.move_finger("0.1,0,0")

        def do_forward(self, args):
            x.move_finger("0,0,0.1")

        def do_back(self, args):
            x.move_finger("0,0,-0.1")

        def do_next_expert_step(self, args):
            print(find_next_step(x, x.target_pos, step_size=0.1, reach_minimum=0.1))

        def do_random(self, args):
            elems = x.clean_split(args, ",")
            if len(elems) <  3:
                print("do_random steps, step_size, dims")
            else:
                finger = len(elems) == 3
                steps = int(elems[0])
                step_size = elems[1]
                dims = int(elems[2])

                for i in range(steps):
                    s = ""
                    for d in range(dims):
                        s += random.choice(["+","-"])
                        s += str(step_size)
                        s += ","
                        if finger:
                            self.do_move_finger(s[:-1])
                        else:
                            self.do_move_cams(s[:-1])
        def do_set_dir(self, args):
            d.set_dir(args)

    prompt = MyPrompt()
    prompt.prompt = '>>> '
    prompt.cmdloop()

# import cv2
import os
class _display:
    def __init__(self, show_obs=True):
        self.show_obs = show_obs
        self.first = True
        self.directory = None

    def set_dir(self, name):
        if not os.path.exists(name):
            os.makedirs(name)
            self.count = 0
            self.directory = name
        else:
            print("directory {} exists".format(name))


    def show(self, env):
        if self.directory is not None:
            self.count += 1
            env.save_cams(self.directory + "/" + str(self.count).zfill(6) + "_")


if __name__ == "__main__":
    env_test(sys.argv[1:])
