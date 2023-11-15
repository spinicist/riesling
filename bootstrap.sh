#!/bin/bash -eu

USAGE="Usage: $0 [options]

Bootstraps the riesling build system (specifies the toolchain for CMake)

Options:
  -f FILE : Use a set of flags for dependency compilation. Currently provided
            options are avx2, abi (for pre-C++11 ABI), and native
"

git submodule update --init --recursive

FLAGS="base"
PAR=""
PREFIX=""
VIEW=""
while getopts "f:hi:j:v" opt; do
    case $opt in
        f) FLAGS="$OPTARG";;
        i) PREFIX="-DCMAKE_INSTALL_PREFIX=$OPTARG -DCMAKE_PREFIX_PATH=$OPTARG";;
        j) export VCPKG_MAX_CONCURRENCY=$OPTARG
           PAR="-j $OPTARG";;
        h) echo "$USAGE"
           return;;
	v) VIEW="-DBUILD_VIEW=ON";;
    esac
done
shift $((OPTIND - 1))

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
  -DFLAGS_FILE="${FLAGS}" \
  $PREFIX $VIEW
cmake --build . $PAR

if [ -n "$PREFIX" ]; then
  cmake --install .
fi
