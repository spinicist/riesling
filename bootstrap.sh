#!/bin/bash -eux

TRIPLET="x64-osx-asan"

vcpkg/bootstrap-vcpkg.sh -useSystemBinaries
cmake -B build -S . \
    -GNinja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake \
    -DVCPKG_TARGET_TRIPLET="$TRIPLET"
cmake --build build
