#!/bin/bash -eux

USAGE="Usage: $0 [options]

Bootstraps the riesling build system (specifies the toolchain for CMake)

Options:
  -i DIR Install riesling to this directory, e.g. $HOME/.local
  -j N   Restrict parallel build to this many threads
  -h     Print this message
"
PAR=""
PREFIX=""
while getopts "f:hi:j:" opt; do
    case $opt in
        i) PREFIX="-DCMAKE_INSTALL_PREFIX=$OPTARG -DCMAKE_PREFIX_PATH=$OPTARG";;
        j) export VCPKG_MAX_CONCURRENCY=$OPTARG
           PAR="-j $OPTARG";;
        h) echo "$USAGE"
           return;;
    esac
done
shift $((OPTIND - 1))

# If vcpkg is not installed, install it
if [ -x "$( command -v vcpkg )" ]; then
  echo "vcpkg installed"
else
  git clone https://github.com/microsoft/vcpkg.git .vcpkg
  cd .vcpkg && ./bootstrap-vcpkg.sh && cd ..
  export VCPKG_ROOT="$PWD/.vcpkg"
  export PATH="$VCPKG_ROOT:$PATH"
fi

# Check for Magick++ and build montage if available
if [ -x "$( command -v Magick++-config )" ]; then
  MONTAGE="-DBUILD_MONTAGE=ON"
else
  MONTAGE=""
fi

mkdir -p build
cmake -S . --preset=default $PREFIX $MONTAGE
cmake --build build $PAR

if [ -n "$PREFIX" ]; then
  cmake --install build
fi
