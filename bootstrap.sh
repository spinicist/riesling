#!/bin/bash -eu

# This script will bootstrap an initial configure & build.
# On Linux and Mac a toolchain will be used which includes -march=native and
# ASan/UBSan support. Other platforms will use the default toolchain.

git submodule update --init --recursive

export VCPKG_OVERLAY_TRIPLETS="$PWD/cmake/triplets"
export FLAGS="base"
while getopts "o" opt; do
    case $opt in
        o) export FLAGS="avx2";;
    esac
done

# Use Ninja if available, otherwise CMake default
if [ -x "$( command -v ninja )" ]; then
  GEN="-GNinja"
else
  GEN=""
fi


mkdir -p build
cd build
cmake -S ../ $GEN \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="cmake/toolchain.cmake"
cmake --build .
