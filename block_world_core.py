#!/usr/bin/env python

import sys
import platform
import matplotlib.pyplot as plt
import subprocess
import requests
import argparse
import io
from PIL import Image
from time import sleep
import numpy as np
import json

class env:
    last_port = 9000

    def __init__(self, log="/dev/stdout", pos_unit=None, rot_unit=None,run = True):
        self.port = env.last_port
        self.uri = "http://localhost:" + str(self.port) + "/"
        env.last_port += 1
        self.pos_unit = pos_unit
        self.rot_unit = rot_unit
        self.log = log

        self.first_connection = True
        if run:
            self.run()

    def run(self):
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

    def do(self, command, args=None):
        not_done = True
        count = 0

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
                print("Timeout on request!!!")
                return None

        if r.status_code != 200:
            print('Status:', r.status_code, 'Problem with the request. Exiting.')
            return None

        return r

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

    def set_params(self, tray_length = None,
                         tray_width = None,
                         tray_height = None,
                         rim_height = None,
                         rim_width = None,
                         obj_min_size = None,
                         obj_max_size = None,
                         max_objects = None,
                         finger_size = None,
                         finger_max_height = None,
                         finger_distance_from_tray = None,
                         target_size = None,
                         stereo_distance = None):

        tray_length = self.round_unit(tray_length, self.pos_unit)
        tray_width = self.round_unit(tray_width, self.pos_unit)
        tray_height = self.round_unit(tray_height, self.pos_unit)
        rim_height = self.round_unit(rim_height, self.pos_unit)
        rim_width = self.round_unit(rim_width, self.pos_unit)
        obj_min_size = self.round_unit(obj_min_size, self.pos_unit)
        obj_max_size = self.round_unit(obj_max_size, self.pos_unit)
        finger_size = self.round_unit(finger_size, self.pos_unit)
        finger_max_height = self.round_unit(finger_max_height, self.pos_unit)
        finger_distance_from_tray = self.round_unit(finger_distance_from_tray, self.pos_unit)
        target_size = self.round_unit(target_size, self.pos_unit)
        stereo_distance = self.round_unit(stereo_distance, self.pos_unit)

        if tray_length is not None:
            self.do("tray_length", str(tray_length))
        if tray_width is not None:
            self.do("tray_width", str(tray_width))
        if tray_height is not None:
            self.do("tray_height", str(tray_height))
        if rim_height is not None:
            self.do("rim_height", str(rim_height))
        if rim_width is not None:
            self.do("rim_width", str(rim_height))
        if obj_min_size is not None:
            self.do("obj_min_size", str(obj_min_size))
        if obj_max_size is not None:
            self.do("obj_max_size", str(obj_max_size))
        if max_objects is not None:
            self.do("max_objects", str(max_objects))
        if finger_size is not None:
            self.do("finger_size", str(finger_size))
        if finger_max_height is not None:
            self.do("finger_max_height", str(finger_max_height))
        if finger_distance_from_tray is not None:
            self.do("finger_distance_from_tray", str(finger_distance_from_tray))
        if target_size is not None:
            self.do("target_size", str(target_size))
        if stereo_distance is not None:
            self.do("stereo_distance", str(stereo_distance))

    def reset(self, **kwargs):
        self.set_params(**kwargs)
        r = self.do("reset")
        self.extract_response(r.json())

    def move_finger(self, inc_pose):
        inc_pose = self.round_pose(inc_pose)
        r = self.do("move_finger", str(inc_pose))
        self.extract_response(r.json())

    def move_cams(self, inc_pose):
        inc_pose = self.round_pose(inc_pose)
        r = self.do("move_cams", str(inc_pose))
        self.extract_response(r.json())

    def get_json_params(self, args):
        r = self.do("get_json_params")
        json_value = r.json()
        print('json respone ' + str(json_value))
        return json_value

    def get_json_objectinfo(self, args):
        r = self.do("get_json_objectinfo")
        json_value = r.json()
        print(json_value)
        return json_value

    def set_json_params(self, args):
        r = self.do("set_json_params", json.dumps(args))

    def reset_json_objectinfo(self, args):
        r = self.do("reset_json_objectinfo", json.dumps(args))
        self.extract_response(r.json())

    def json_vector3(self, vec3):
        new_v = dict()
        new_v['x'] = float(vec3[0])
        new_v['y'] = float(vec3[1])
        new_v['z'] = float(vec3[2])
        return new_v

    def json_quaternion(self, q):
        new_q = dict()
        new_q['x'] = float(q[0])
        new_q['y'] = float(q[1])
        new_q['z'] = float(q[2])
        new_q['w'] = float(q[4])
        return new_q

    def json_transform(self, position, rotation, localScale):
        new_t = dict()
        new_t['postion'] = self.json_vector3(position)
        new_t['rotation'] = self.json_quaternion(rotation)
        new_t['localScale'] = self.json_vector3(localScale)
        return new_t

    def quaternion_id(self):
        return [0, 0, 0, 1]

    def json_objectinfo(self,
                          finger_size,
                          finger_position,
                          target_size,
                          target_position,
                          camera_position,
                          camera_rotation):
        q_id = self.quaternion_id()
        j = dict()
        j['ObjList'] = list()
        j['finger'] = self.json_transform(finger_position, q_id, [finger_size, finger_size, finger_size])
        j['target'] = self.json_transform(target_position, q_id, [target_size, target_size, target_size])
        j['main_camera'] = self.json_transform(camera_position, camera_rotation, [1, 1, 1])

        return j

    def extract_cam(self, data, key):
        if key in data:
            cam_stream = io.BytesIO(bytearray(data[key]))
            return Image.open(cam_stream)
        else:
            return None

    def extract_float(self, data, key):
        if key in data:
            return float(data[key])
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

    def extract_response(self, data):
        self.cam1 = self.extract_cam(data, "cam1")
        self.cam2 = self.extract_cam(data, "cam2")
        self.target_pos = self.extract_vector3(data, "target_pos")
        self.finger_pos = self.extract_vector3(data, "finger_pos")
        self.target_rot = self.extract_vector3(data, "target_rot")
        self.finger_rot = self.extract_vector3(data, "finger_rot")
        self.collision = self.extract_bool(data, "collision")

    def clean_split(self, s, delim):
        s = s.replace(" ", "")
        s = s.replace("(", "")
        s = s.replace(")", "")

        return s.split(delim)

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
    parser.set_defaults(reset=True)
    parser.set_defaults(no_units=False)
    parser.set_defaults(show_obs=False)
    parser.set_defaults(run=True)
    cmd_args = parser.parse_args(argv)

    if cmd_args.no_units:
        cmd_args.pos_unit = None
        cmd_args.rot_unit = None

    x = env(pos_unit=cmd_args.pos_unit, rot_unit=cmd_args.rot_unit, run=cmd_args.run)
    d = _display(show_obs=cmd_args.show_obs)

    if cmd_args.reset:
        x.reset(tray_length=3.0, tray_width=2.0, stereo_distance=0.5)
        d.show(x)

    var_dict = dict()

    from cmd import Cmd
    import random

    class MyPrompt(Cmd):
        def do_quit(self, args):
            x.quit()
            raise SystemExit

        def do_EOF(self, args):
            x.quit()
            print("")
            raise SystemExit

        def do_reset(self, args):
            if args:
                x.reset(max_objects=int(args))
            else:
                x.reset(tray_length=3.0, tray_width=2.0, stereo_distance=0.5, max_objects=20)
            d.show(x)

        def do_get_json_params(self, args):
            json_value = x.get_json_params(args)
            if args:
                var_dict[args] = json_value

        def do_get_json_objectinfo(self, args):
            json_value = x.get_json_objectinfo(args)
            if args:
                var_dict[args] = json_value

        def do_set_json_params(self, args):
            if args in var_dict:
                x.set_json_params(var_dict[args])
            else:
                x.set_json_params(args)

        def do_reset_json_objectinfo(self, args):
            if args in var_dict:
                x.reset_json_objectinfo(var_dict[args])
            else:
                x.reset_json_objectinfo(args)
            d.show(x)

        def do_move_finger(self, args):
            x.move_finger(args)
            d.show(x)

        def do_move_cams(self, args):
            x.move_cams(args)
            d.show(x)

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

    prompt = MyPrompt()
    prompt.prompt = '>>> '
    prompt.cmdloop()

class _display:
    def __init__(self, show_obs=True):
        self.show_obs = show_obs
        self.first = True

        if self.show_obs:
            f, (self.ax1, self.ax2, self.ax3) = plt.subplots(3)
            plt.ion()
            self.ax1 = plt.subplot2grid((2, 2), (0, 0))
            self.ax2 = plt.subplot2grid((2, 2), (0, 1))
            self.ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
            self.ax1.axis('off')
            self.ax2.axis('off')
            self.ax3.axis('off')

    def show(self, env):
        if self.show_obs:
            if env.collision:
                print("BOOM")
            self.ax1.imshow(env.cam1)
            self.ax2.imshow(env.cam2)
            if env.collision:
                self.ax3.text(0.5, 0.5, 'BOOM', horizontalalignment='center', verticalalignment='center')
            plt.draw()
            if self.first:
                self.first = False
                plt.show()

if __name__ == "__main__":
    env_test(sys.argv[1:])
