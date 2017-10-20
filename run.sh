#!/usr/bin/env bash
cwd=`pwd`
if [ "$(uname)" == "Darwin" ]; then
    run="builds/osx_player.app/Contents/MacOS/osx_player"
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    echo dont know how to run unity on linux
elif [ "$(expr substr $(uname -s) 1 10)" == "MINGW32_NT" ]; then
    # Do something under 32 bits Windows NT platform
    echo dont know how to run unity on Windows 32 bit
elif [ "$(expr substr $(uname -s) 1 10)" == "MINGW64_NT" ]; then
    echo dont know how to run unity on Windows 64 bit
fi

$run  -logFile /dev/stdout -screen-width 640 -screen-height 480 -screen-quality ultra -screen-fullscreen 0
