#!/usr/bin/env python

import os.path
import subprocess
import shlex
import platform
import tempfile
import re
import sys


def pci_records():
    records = []
    command = shlex.split('lspci -vmm')
    output = subprocess.check_output(command).decode()

    for devices in output.strip().split("\n\n"):
        record = {}
        records.append(record)
        for row in devices.split("\n"):
            key, value = row.split("\t")
            record[key.split(':')[0]] = value

    return records

def generate_xorg_conf(devices):
    xorg_conf = []

    device_section = """
Section "Device"
    Identifier     "Device{device_id}"
    Driver         "nvidia"
    VendorName     "NVIDIA Corporation"
    BusID          "{bus_id}"
EndSection
"""
    server_layout_section = """
Section "ServerLayout"
    Identifier     "Layout0"
    {screen_records}
EndSection
"""
    screen_section = """
Section "Screen"
    Identifier     "Screen{screen_id}"
    Device         "Device{device_id}"
    DefaultDepth    24
    Option         "AllowEmptyInitialConfiguration" "True"
    SubSection     "Display"
        Depth       24
        Virtual 1024 768
    EndSubSection
EndSection
"""
    screen_records = []
    for i, device in enumerate(devices):
        bus_id = 'PCI:' + ':'.join(map(lambda x: str(int(x, 16)), re.split(r'[:\.]', device['Slot'])))
        xorg_conf.append(device_section.format(device_id=i, bus_id=bus_id))
        xorg_conf.append(screen_section.format(device_id=i, screen_id=i))
        screen_records.append('Screen {screen_id} "Screen{screen_id}" 0 0'.format(screen_id=i))
    
    xorg_conf.append(server_layout_section.format(screen_records="\n    ".join(screen_records)))

    return "\n".join(xorg_conf)

def startx(display=15):
    if platform.system() != 'Linux':
        raise Exception("Can only run startx on linux")
    records = list(filter(lambda r: r.get('Vendor', '') == 'NVIDIA Corporation' and (r['Class'] == 'VGA compatible controller' or r['Class'] == '3D controller'), pci_records()))

    if not records:
        raise Exception("no nvidia cards found")

    try:
        fd, path = tempfile.mkstemp()
        with open(path, "w") as f:
            f.write(generate_xorg_conf(records))
        command = shlex.split("sudo Xorg -noreset +extension GLX +extension RANDR +extension RENDER -config %s :%s" % (path, display))
        subprocess.call(command)
    finally: 
        os.close(fd)
        os.unlink(path)

if __name__ == "__main__":
    argv = sys.argv
    if len(argv) == 1:
        disp = 15
    else:
        disp = int(argv[1])

    print("starting X on display {}".format(disp))

    startx(disp)

    print("export DISPLAY=:{}.0", disp)
