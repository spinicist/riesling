# Risling Python interface

## `data`

## `plot`

## `command`

The `riesling.command` module includes a `riesling` function which can be used to execute riesling commands with python objects as input and output. It works in much the same way as the BART python interface.

Main way to execute is

```
out = riesling('my_command <arguments>', main_file, additional_file, flag_with_data=data1, flat_with_data2=data2)
```

1. The first argument is the riesling command, like `admm --l1=0.01`. 
2. The second argument should be the main data object, like your k-space data. This should be a python object, given by the `riesling.data.read` command.
3. Additional positional arguments are additional data needed by the command which are not keyword arguments.
4. Additional keyword arguments are used when additional input data is needed, like sense maps. Add these with the long keyword argument and the python object.
5. The output `out` is a dictionary with keys corresponding to the suffix used in the output by the command, this is typically the name of the command. It will also output the `stdout` and `stderr`.

The command will write the python objects to temp files and remove them after executing the command. It assumes that you have riesling in your path.

Examples of how to use it

```python
# Load input data
my_data = riesling.data.read('my_kspace_data.h5')

# Make sense maps
out = riesling('sense', my_data)
sense_maps = out['sense']

# Run admm
out = riesling('admm --tgv=0.01 --sdc=pipe', my_data, sense=sense_maps)
tgv_img = out['admm']
```