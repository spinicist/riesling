![Logo](riesling-logo.png)

[![Build](https://github.com/spinicist/riesling/workflows/Build/badge.svg)](https://github.com/spinicist/QUIT/actions)

## Radial Interstices Enable Speedy Low-Volume imagING

This is a reconstruction toolbox optimised for 3D ZTE MR images. While there are many high quality MR recon toolboxes, e.g. [BART](http://mrirecon.github.io/bart/), available, these are mostly optimised for 2D sequences. 3D non-cartesian sequences present unique challenges for efficient reconstruction, so we wrote our own.

We have submitted this toolbox to ISMRM 2020.

## Authors

Tobias C Wood & Emil Ljungberg

## Installation

Pre-compiled executables are provided for Linux and Mac OS X in a .tar.gz 
archive from http://github.com/spinicist/riesling/releases. Download the archive and 
extract it with `tar -xzf riesling-platform.tar.gz`. Then, move the resulting `riesling` executable to somewhere on your `$PATH`, for instance `/usr/local/bin`. That's it.

- MacOS Catalina or higher users should use `curl` to download the binary, i.e. 
  type `curl -L https://github.com/spinicist/QUIT/releases/download/v1.0/riesling-macos.tar.gz`
  This is because Safari now sets the quarantine attribute of all downloads,
  which prevents them being run as the binary is unsigned. It is possible to 
  remove the quarantine flag with `xattr`, but downloading with `curl` is more 
  straightforward.
- The Linux executable is compiled on Ubuntu 16.04 with GLIBC version 2.3 and a 
  statically linked libc++. This means it will hopefully run on most modern 
  Linux distributions. Let me know if it doesn't.

## Compilation

If you wish to compile RIESLING yourself, compilation should hopefully be straightforward. RIESLING relies on `vcpkg` for dependency management. To download and compile RIESLING, follow these steps:

1. Clone the repository. `git clone https://github.com/spinicist/riesling`
2. Run `git submodule init` to clone a copy of the `vcpkg` repo.
3. Edit `bootstrap.sh` to choose your target triplet.
4. Run `bootstraph.sh`.

## Usage

RIESLING comes as a single executable file with multiple commands, similar to `git` or `bart`. Type `riesling` to see a list of all the available commands.

RIESLING uses HDF5 (.h5) files as an input/intermediate format and NIFTI (.nii) as a final output. To create an example digital phantom, use `riesling phantom`. RIESLING will append suffixes to input filenames when writing outputs to indicate which command was executed.

There are three reconstruction algorithms currently provided in RIESLING - plain root-sum-squares (`riesling rss`), iterative cgSENSE with Töplitz embedding (`riesling toe`), and iterative recon with TGV regularization (`riesling tgv`).

## Getting Help

If you can't find an answer to a problem in the documentation or help strings, 
you can open an [issue](https://github.com/spinicist/riesling/issues), or find the developers on Twitter ([@spinicist](https://twitter.com/spinicist)), or in the #includecpp Discord.
