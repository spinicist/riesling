![Logo](riesling-logo.png)

[![Build](https://github.com/spinicist/riesling/workflows/Build/badge.svg)](https://github.com/spinicist/riesling/actions)
[![DOI](https://zenodo.org/badge/317237623.svg)](https://zenodo.org/badge/latestdoi/317237623)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.03500/status.svg)](https://doi.org/10.21105/joss.03500)

## Radial Interstices Enable Speedy Low-Volume imagING

This is a reconstruction toolbox optimised for 3D ZTE MR images. There are many high quality MR recon toolboxes available, e.g. [BART](http://mrirecon.github.io/bart/), but these are mostly optimised for 2D sequences. 3D non-cartesian sequences present unique challenges for efficient reconstruction, so we wrote our own.

This toolbox was presented at ISMRM 2020.

## Authors

Tobias C Wood, Emil Ljungberg, Florian Wiesinger.

## Installation

Pre-compiled executables are provided for Linux and Mac OS X in a .tar.gz 
archive from http://github.com/spinicist/riesling/releases. Download the 
archive and extract it with `tar -xzf riesling-platform.tar.gz`. Then, move the 
resulting `riesling` executable to somewhere on your `$PATH`, for instance 
`/usr/local/bin`. That's it.

- MacOS Catalina or higher users should use `curl` to download the binary, i.e. 
  `curl -L https://github.com/spinicist/riesling/releases/download/v1.0/riesling-macos.tar.gz`. 
  This is because Safari now sets the quarantine attribute of all downloads, 
  which prevents them being run as the binary is unsigned. It is possible to 
  remove the quarantine flag with `xattr`, but downloading with `curl` is more 
  straightforward.
- The Linux executable is compiled on Ubuntu 18.04 and a statically linked
  libstdc++. This means it will hopefully run on most modern Linux
  distributions. Let us know if it doesn't.
- The Mac executable is compiled with MacOS 11. GitHub CI uses Intel machines,
  and a native M1 version will not be available until these changes.

## Compilation

If you wish to compile RIESLING yourself, compilation should hopefully be 
straightforward as long as you have access to a C++20 compiler (GCC 10 or
higher, Clang 7 or higher). RIESLING relies on `vcpkg` for dependency
management. To download and compile RIESLING, follow these steps:

### 0. MacOS Dependencies
Install the [MacOS vcpkg dependencies](https://github.com/microsoft/vcpkg#installing-macos-developer-tools).

1. XCode from the AppStore
2. Run `$ xcode-select --install` in the terminal

You may also need to install `pkg-config` depending on your macOS version. This
is easily installed with [Homebrew](https://brew.sh/) using
```
$ brew install pkg-config
```

Apple Silicon (M1) is now supported.

### 0. Linux Dependencies
Install the [Linux vcpkg dependencies](https://github.com/microsoft/vcpkg#installing-linux-developer-tools).
These include `cmake`, `tar`, `curl`, `zip`, `unzip`, `pkg-config` \&
`build-essential`. You may be surprised by which distributions do not include
these by default.

### 1. Clone repository
```
$ git clone https://github.com/spinicist/riesling
```

### 2. Compile
In the `riesling` folder execute
```
$ ./bootstrap.sh
```

## Usage

RIESLING comes as a single executable file with multiple commands, similar to 
`git` or `bart`. Type `riesling` to see a list of all the available commands. If
you run a RIESLING command without any additional parameter RIESLING will output
all available options for the given command.

RIESLING uses HDF5 (.h5) files but can also output NIFTI (.nii). To create an 
example digital phantom, use `riesling phantom`. RIESLING will append suffixes 
to input filenames when writing outputs to indicate which command was executed.

There are several reconstruction algorithms currently provided in RIESLING. 
Simple non-iterative recon is available with (`riesling recon`).

A separate examples repository https://github.com/spinicist/riesling-examples
contains Jupyter notebooks demonstrating most functionality. These can also be
run on the mybinder website.

## Documentation & Help

Documentation is available at https://riesling.readthedocs.io.

If you can't find an answer there or in the help strings, 
you can open an [issue](https://github.com/spinicist/riesling/issues), or find
the developers on Twitter ([@spinicist](https://twitter.com/spinicist)) or
e-mail tobias.wood@kcl.ac.uk.
