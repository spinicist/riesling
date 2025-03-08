#!/usr/bin/env python3

"""
Script to run MERLIN registration on the command line. 

.. code:: bash

    usage: merlin <command> [<args>]

    Available commands are:
        reg         Register navigator series
        report      View report of data
        metric      Image metric analysis
        animation   Navigator and registration animation
        ssim        Calculate Structural Similarity Index Measure
        aes         Calculate Average Edge Strength
        nrmse       Calculate Normalised Root Mean Squared Error

To get more help for a specific command add ``-h``.

"""

from builtins import ValueError
import argparse
import logging
import os
import pickle
import sys

import h5py
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from . import io
from .merlin import versor3D_registration, histogram_threshold_estimator


class Merlin_parser(object):
    """
    Class to produce a subcommand arg parser like git for MERLIN

    Inspired by: https://chase-seibert.github.io/blog/2014/03/21/python-multilevel-argparse.html
    """

    def __init__(self):
        parser = argparse.ArgumentParser(description='MERLIN Python tools',
                                         usage='''pymerlin <command> [<args>]

    Available commands are:
        reg         Register navigator series
    '''
                                         )

        parser.add_argument('command', help='Subcommand to run')
        args = parser.parse_args(sys.argv[1:2])

        # Here we check if out object (the class) has a function with the given name
        if not hasattr(self, args.command):
            print('Unrecognized command')
            parser.print_help()
            exit(1)

        # Call the method
        getattr(self, args.command)()

    def reg(self):
        parser = argparse.ArgumentParser(description='MERLIN Registration',
                                         usage='merlin reg [<args>] [versor3D_registration arguments]')

        parser.add_argument("--fixed", help="Fixed image",
                            required=True, type=arg_check_h5)
        parser.add_argument("--moving", help="Moving image",
                            required=True, type=arg_check_h5)
        parser.add_argument(
            "--fixed_mask", help="Fixed space mask", required=False, type=arg_check_h5)
        parser.add_argument(
            "--reg", help="Registration parameters", required=False, type=str, default=None)
        parser.add_argument("--log", help="Registration history log",
                            required=False, type=str, default=None)
        parser.add_argument(
            "--fixout", help="Name of fixed image output", required=False, type=arg_check_nii)
        parser.add_argument(
            "--moveout", help="Name of registered moving image output", required=False, type=arg_check_nii)
        parser.add_argument("--sigma", help="List of sigmas",
                            required=False, default=[0], nargs="+", type=int)
        parser.add_argument("--shrink", help="Shrink factors",
                            required=False, default=[1], nargs="+", type=int)
        parser.add_argument("--metric", help="Image metric",
                            required=False, default="MS")
        parser.add_argument(
            "--verbose", help="Log level (0,1,2)", default=2, type=int)
        # remember to remove
        parser.add_argument('--bad-frames', type=str,
                            help='Comma-separated list of frame indices to remove')

        # Since we are inside the subcommand now we skip the first two
        # arguments on the command line
        args, unknown_args = parser.parse_known_args(sys.argv[2:])
        main_reg(args, unknown_args)

def main_reg(args, unknown_args):
    if not args.reg:
        fix_base = os.path.splitext(os.path.basename(args.fixed))
        move_base = os.path.splitext(os.path.basename(args.moving))
        reg_name = "{}_2_{}_reg.p".format(move_base, fix_base)
    else:
        reg_name = args.reg

    more_args = {}
    i = 0
    while i < len(unknown_args):
        k = unknown_args[i].split('--')[1]
        val = unknown_args[i+1]
        i += 2
        try:
            more_args[k] = float(val)
        except ValueError:
            more_args[k] = val

    r, rout, reg_fname = versor3D_registration(fixed_image_fname=args.fixed,
                                               moving_image_fname=args.moving,
                                               moco_output_name=args.moveout,
                                               fixed_mask_fname=args.fixed_mask,
                                               fixed_output_name=args.fixout,
                                               reg_par_name=reg_name,
                                               iteration_log_fname=args.log,
                                               sigmas=args.sigma,
                                               shrink=args.shrink,
                                               metric=args.metric,
                                               verbose=args.verbose,
                                               **more_args)

def main():
    # Everything is executed by initialising this class.
    # The command line arguments will be parsed and the appropriate function will be called
    Merlin_parser()


if __name__ == '__main__':
    main()
