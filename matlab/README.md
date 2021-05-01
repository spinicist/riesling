# MATLAB Integration
Includes two main functions for reading and writing riesling k-space data

```
[data, traj, info, meta] = read_riesling(fname);
```

```
write_riesling(fname, data, traj, info, meta);
```

## Examples
Two examples are included:

1. `riesling_matlab_demo.m`: Demonstrating basic data input, output, manipulation and plotting.
2. Combining BART with riesling (requires working BART installation)
