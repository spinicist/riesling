#!/bin/bash -eux

git submodule update --init --recursive
cmake -B build -S . \
    -GNinja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE=cmake/Sanitizers.cmake
cmake --build build
