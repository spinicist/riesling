#!/bin/bash -eu

# This script will bootstrap an initial configure & build.
# On Linux and Mac a toolchain will be used which includes -march=native and
# ASan/UBSan support. Other platforms will use the default toolchain.

git submodule update --init --recursive

export FLAGS="base"
while getopts "a" opt; do
    case $opt in
        a) export FLAGS="avx2";;
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
  -DCMAKE_TOOLCHAIN_FILE="cmake/toolchain.cmake" \
  -DFLAGS_FILE="${FLAGS}"
cmake --build .
