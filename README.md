![Logo](riesling-logo.png)

[![Build](https://github.com/spinicist/riesling/workflows/Build/badge.svg)](https://github.com/spinicist/riesling/actions)

## Radial Interstices Enable Speedy Low-Volume imagING

This is a reconstruction toolbox optimised for 3D ZTE MR images. There are many high quality MR recon toolboxes available, e.g. [BART](http://mrirecon.github.io/bart/), but these are mostly optimised for 2D sequences. 3D non-cartesian sequences present unique challenges for efficient reconstruction, so we wrote our own.

This toolbox was presented at ISMRM 2020.

## Authors

Tobias C Wood, Emil Ljungberg, Florian Wiesinger.

## Installation

Pre-compiled executables are provided for Linux and Mac OS X in a .tar.gz 
archive from http://github.com/spinicist/riesling/releases. Download the 
archive and extract it with `tar -xzf riesling-platform.tar.gz`. Then, move the ]
resulting `riesling` executable to somewhere on your `$PATH`, for instance 
`/usr/local/bin`. That's it.

- MacOS Catalina or higher users should use `curl` to download the binary, i.e. 
  `curl -L https://github.com/spinicist/riesling/releases/download/v1.0/riesling-macos.tar.gz`. 
  This is because Safari now sets the quarantine attribute of all downloads, 
  which prevents them being run as the binary is unsigned. It is possible to 
  remove the quarantine flag with `xattr`, but downloading with `curl` is more 
  straightforward.
- The Linux executable is compiled on Ubuntu 16.04 with GLIBC version 2.3 and a 
  statically linked libc++. This means it will hopefully run on most modern 
  Linux distributions. Let us know if it doesn't.

## Compilation

If you wish to compile RIESLING yourself, compilation should hopefully be 
straightforward as long as you have access to a C++17 compiler (GCC 8 or higher,
Clang 7 or higher). RIESLING relies on `vcpkg` for dependency management. To 
download and compile RIESLING, follow these steps:

0. Install the dependencies: `cmake` \& `curl` (`vcpkg` requires `curl`).
1. Clone the repository. `git clone https://github.com/spinicist/riesling`
2. Run `bootstraph.sh`.

## Usage

RIESLING comes as a single executable file with multiple commands, similar to 
`git` or `bart`. Type `riesling` to see a list of all the available commands. If you run a RIESLING command without any additional parameter RIESLING will output all available options for the given command.

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
