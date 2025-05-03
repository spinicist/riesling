# This code is heavily inspired/copied from BART python/bart.py
# 
# BART License statement
# --------------------------------------------------------------------------------
# Copyright (c) 2013-2018. The Regents of the University of California.
# Copyright (c) 2013-2022. BART Developer Team and Contributors.
# Copyright (c) 2012. Intel Corporation. (src/lapacke/)
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 
# License statement from bart.py
# Copyright 2016. The Regents of the University of California.
# All rights reserved. Use of this source code is governed by 
# a BSD-style license which can be found in the LICENSE file.
#
# Authors: 
# 2016 Siddharth Iyer <sid8795@gmail.com>
# 2018 Soumick Chatterjee <soumick.chatterjee@ovgu.de> , WSL Support
# ----------------------------------------------------------------------------------

import subprocess as sp
import tempfile as tmp
import os
from .data import write, read
from glob import glob


def riesling(cmd, *args, **kwargs):
    """riesling

    meant to be run like
    riesling('admm --llr=2', file, sense=my_sense_data)

    - The first command is a command line string, equivalent to the riesling command
    - The positional arguments are the same positional arguments that are passed to riesling
    typically only one variable. Here they are passed as python objects, unless they are given as strings, then
    they are used like files. The first 
    - The keyword arguments are used for when there are keyword arguments used to pass in file information, 
    like a sense map or a basis file. If these are passed in as a string for an existing file, then they can be passed
    in with the command string, if you want to provide a python object, use the keyword argument and specify the 
    python object you want to pass in.

    Args:
        cmd (_type_): _description_
        *args: Input files
        **kwargs: Optional arguments

    Returns:
        _type_: _description_
    """
    
    # Assume riesling is in path
    riesling_path = ''

    rm_files = []

    # Base name for input
    in_bname = tmp.NamedTemporaryFile().name
    nargin = len(args)

    # First input is the one that decides the base name for the output
    main_file = in_bname+'.h5'
    write(filename=main_file, data=args[0], data_type=args[0].name)
    rm_files.append(main_file)

    # Additional input files
    add_infiles = [in_bname + 'in' + str(idx) for idx in range(nargin-1)]

    for idx in range(nargin-1):
        write(filename=add_infiles[idx], data=args[idx+1], data_type=args[idx].name)
        rm_files.append(add_infiles[idx])

    cmd = cmd.split(" ")

    # Read keyword arguments for additional input files
    for k in kwargs.keys():
        fname = f"{in_bname}KWARG{k}.h5"
        write(filename=fname, data=kwargs[k], data_type=kwargs[k].name)
        rm_files.append(fname)
        cmd.append(f'--{k}={fname}')

    shell_cmd = [os.path.join(riesling_path, 'riesling'), *cmd, f'--out={in_bname}', main_file, *add_infiles]

    # run bart command
    ERR, stdout, stderr = execute_cmd(shell_cmd)

    # Find the output files
    files = glob(f"{in_bname}-*.h5")
    file_path = os.path.split(files[0])[0]
    file_base = os.path.split(files[0])[1].split('-')[0]
    keys = [os.path.split(f)[-1].split('-')[1].split('.h5')[0] for f in files]

    output = {'ERR':ERR, 'stdout':stdout, 'stderr':stderr}
    for k in keys:
        fname = os.path.join(file_path, f"{file_base}-{k}.h5")
        output[k] = read(fname)
        rm_files.append(fname)

    for f in rm_files:
        os.remove(f)

    if ERR:
        print(f"Command exited with error code {ERR}.")
        return

    return output


def execute_cmd(cmd):
    """
    Execute a command in a shell.
    Print and catch the output.
    """
    
    errcode = 0
    stdout = ""
    stderr = ""

    # remove empty strings from cmd
    cmd = [item for item in cmd if len(item)]

    # execute cmd
    proc = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.PIPE, universal_newlines=True)

    # print to stdout
    for stdout_line in iter(proc.stdout.readline, ""):
        stdout += stdout_line
        print(stdout_line, end="")
    proc.stdout.close()

    # in case of error, print to stderr
    errcode = proc.wait()
    if errcode:
        stderr = "".join(proc.stderr.readlines())
        print(stderr)
    proc.stderr.close()

    return errcode, stdout, stderr