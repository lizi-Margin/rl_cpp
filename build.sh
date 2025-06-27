#!/bin/bash

mkdir -p build
cd build || exit 1

cmake \
  -DCMAKE_PREFIX_PATH='/opt/libtorch-cpu;/opt/cuda' \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
  .. && make -j18
cd ..

if [ "$1" = "run" ]; then
  ./build/rl
fi
