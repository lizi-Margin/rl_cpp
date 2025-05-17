#!/bin/bash

mkdir -p build
cd build || exit 1

export CPATH="$CPATH"":/opt/libtorch/include/"
export CPATH="$CPATH"":/opt/libtorch/include/torch/csrc/api/include/"

cmake \
  -DCMAKE_PREFIX_PATH=/opt/libtorch \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
  .. && make -j12
cd ..

./build/rl