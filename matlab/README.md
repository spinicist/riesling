# MATLAB Integration
Includes two main functions for reading and writing riesling k-space data

```
varargout = riesling_read(fname);
```

```
riesling_write(fname, data, traj, info, meta);
```

## Info struct

You can create an empty header info struct with

info = riesling_info();

## Examples
One examples is included:

1. `riesling_matlab_demo.m`: Demonstrating basic data input, output, manipulation and plotting.
