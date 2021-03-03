![Logo](riesling-logo.png)

[![Build](https://github.com/spinicist/riesling/workflows/Build/badge.svg)](https://github.com/spinicist/riesling/actions)

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

0. Install the dependencies: `cmake` \& `curl` (`vcpkg` requires `curl`).
1. Clone the repository. `git clone https://github.com/spinicist/riesling`
2. Run `bootstraph.sh`.


## Usage

RIESLING comes as a single executable file with multiple commands, similar to `git` or `bart`. Type `riesling` to see a list of all the available commands.

RIESLING uses HDF5 (.h5) files as an input/intermediate format and NIFTI (.nii) as a final output. To create an example digital phantom, use `riesling phantom`. RIESLING will append suffixes to input filenames when writing outputs to indicate which command was executed.

There are several reconstruction algorithms currently provided in RIESLING - simple non-iterative recon with root-sum-squares channel combination (`riesling rss`), non-iterative recon with self-calibrating sensitivy map extraction (`riesling sense`), iterative cgSENSE with Töplitz embedding (`riesling cg`), and iterative recon with TGV regularization (`riesling tgv`). An experimental version of the Non-Uniform Fourier Transform/Direct Summation method (`riesling ds`) is also provided, but not recommended except for curiosity's sake. Coil compression via PCA/SVD is also provided (`riesling compress`).

A demo using the dataset from the [ISMRM CG-SENSE Reproducibility challenge](https://ismrm.github.io/rrsg/challenge_one/) can be found in `examples/rrsg_cgsense`, which includes both conversion to the RIESLING .h5 format and examples of how to run RIESLING.

## Input Format

The RIESLING input .h5 is similar but simpler than ISMRMRD (ISMRM raw data) format. The .h5 file should have three entries: `info` - the header information, `traj` - the trajectory, and a group called `data`, within which each k-space volume is written with a four-digit zero-padded identifier. In other words, it should look like this, where `Nr` is the number of read-out points along a spoke, `Ns` is the number of spokes, and `Nc` is the number of channels:

```
/info - Header struct (see below)
/traj - Trajectory dataset specified as a (3, Nr, Ns) float matrix
/data - Group
-/0000 - k-Space dataset for volume 0 as an (Nc, Nr, Ns) complex float matrix
```

Note that the k-space data ordering is channel-first rather than channel-last. A possible definition of the header structure in plain C++ is as follows. The real definition can be found in `info.h` and uses Eigen types:

```
struct Info
{
  long matrix[3];
  float voxel_size[3];
  long read_points;
  long read_gap;
  long spokes_hi;
  long spokes_lo;
  float lo_scale;
  long channels;
  long volumes = 1;
  float tr = 1.f;
  float origin[3];
  float direction[9];
}
```

The output of `h5dump` on an example phantom dataset is as follows:

```
HDF5 "tom.h5" {
GROUP "/" {
   GROUP "data" {
      DATASET "0000" {
         DATATYPE  H5T_COMPOUND {
            H5T_IEEE_F32LE "r";
            H5T_IEEE_F32LE "i";
         }
         DATASPACE  SIMPLE { ( 12, 64, 4096 ) / ( 12, 64, 4096 ) }
         DATA {h5dump error: unable to print data

         }
      }
   }
   DATASET "info" {
      DATATYPE  H5T_COMPOUND {
         H5T_ARRAY { [3] H5T_STD_I64LE } "matrix";
         H5T_ARRAY { [3] H5T_IEEE_F32LE } "voxel_size";
         H5T_STD_I64LE "read_points";
         H5T_STD_I64LE "read_gap";
         H5T_STD_I64LE "spokes_hi";
         H5T_STD_I64LE "spokes_lo";
         H5T_IEEE_F32LE "lo_scale";
         H5T_STD_I64LE "channels";
         H5T_STD_I64LE "volumes";
         H5T_IEEE_F32LE "tr";
         H5T_ARRAY { [3] H5T_IEEE_F32LE } "origin";
         H5T_ARRAY { [9] H5T_IEEE_F32LE } "direction";
      }
      DATASPACE  SIMPLE { ( 1 ) / ( 1 ) }
      DATA {
      (0): {
            [ 64, 64, 64 ],
            [ 3.75, 3.75, 3.75 ],
            64,
            0,
            4096,
            0,
            1,
            12,
            1,
            1,
            [ 0, 0, 0 ],
            [ 1, 0, 0, 0, 1, 0, 0, 0, 1 ]
         }
      }
   }
   DATASET "traj" {
      DATATYPE  H5T_IEEE_F32LE
      DATASPACE  SIMPLE { ( 3, 64, 4096 ) / ( 3, 64, 4096 ) }
      DATA {h5dump error: unable to print data

      }
   }
}
}
```

## Getting Help

If you can't find an answer to a problem in the documentation or help strings, 
you can open an [issue](https://github.com/spinicist/riesling/issues), or find the developers on Twitter ([@spinicist](https://twitter.com/spinicist)), or in the #includecpp Discord.
