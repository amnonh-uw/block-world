#!/usr/bin/env bash
cwd=`pwd`
if [ "$(uname)" == "Darwin" ]; then
    unity='/Applications/Unity/Unity.app/Contents/MacOS/Unity'
    unity='open -W /Applications/Unity/Unity.app --args'
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    echo dont know how to run unity on linux
elif [ "$(expr substr $(uname -s) 1 10)" == "MINGW32_NT" ]; then
    # Do something under 32 bits Windows NT platform
    echo dont know how to run unity on Windows 32 bit
elif [ "$(expr substr $(uname -s) 1 10)" == "MINGW64_NT" ]; then
    echo dont know how to run unity on Windows 64 bit
fi
#batchmode="-batchmode"
batchmode=""
echo 'Building linux player'
$unity -quit $batchmode -logFile /dev/stdout -projectPath $cwd/block_world -buildLinuxUniversalPlayer $cwd/builds/linux_player
echo 'Building mac player'
$unity -quit $batchmode -logFile /dev/stdout -projectPath $cwd/block_world -buildOSXPlayer $cwd/builds/osx_player

