#!/bin/bash -e

USAGE="Usage: $0 [options]

Bootstraps the riesling build system (specifies the toolchain for CMake)

Options:
  -g     Build GEWURZ
  -i DIR Install riesling to this directory, e.g. $HOME/.local
  -j N   Restrict parallel build to this many threads
  -h     Print this message
  -w     Add warnings
"
PAR=""
PREFIX=""
GEWURZ="-DBUILD_GEWURZ=OFF"
ALL=""
DEBUG=""
RELDBG="-O2 -g -DNDEBUG"
while getopts "ghi:j:w" opt; do
    case $opt in
        g) GEWURZ="-DBUILD_GEWURZ=ON";;
        i) PREFIX="-DCMAKE_INSTALL_PREFIX=$OPTARG";;
        j) export VCPKG_MAX_CONCURRENCY=$OPTARG
           PAR="-j $OPTARG";;
        h) echo "$USAGE"
           return;;
        w) ALL="-Wall -Wshadow"
           DEBUG="-g -fsanitize=address,undefined"
           RELDBG="-O2 -g -DNDEBUG -fsanitize=address,undefined";;
    esac
done
shift $((OPTIND - 1))

# Use Ninja if available, otherwise CMake default
if [ -x "$( command -v ninja )" ]; then
  GEN="-GNinja"
else
  GEN=""
fi

# If vcpkg is not installed, install it
if [[ (-x "$( command -v vcpkg )") && (-n $VCPKG_ROOT) ]]; then
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
cmake -S . -B build $GEN\
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" \
  -DVCPKG_INSTALL_OPTIONS="--no-print-usage" \
  -DCMAKE_CXX_FLAGS="$ALL" -DCMAKE_CXX_FLAGS_DEBUG="$DEBUG" -DCMAKE_CXX_FLAGS_RELWITHDEBINFO="$RELDBG"\
  "$PREFIX" "$MONTAGE" "$GEWURZ"

cmake --build build $PAR

if [ -n "$PREFIX" ]; then
  cmake --install build
fi
