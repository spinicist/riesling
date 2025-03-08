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
import sys
import itk
import h5py
import numpy as np

from . import io
from .merlin import create_itk_image, versor3D_registration, histogram_threshold_estimator


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

        parser.add_argument("input", help="Input navigator images", type=str)
        parser.add_argument("output", help="Output transforms file", type=str)
        parser.add_argument("--mask", help="Navigator mask",
                            required=False, type=str)
        parser.add_argument("--sigma", help="List of sigmas",
                            required=False, default=[0], nargs="+", type=int)
        parser.add_argument("--shrink", help="Shrink factors",
                            required=False, default=[1], nargs="+", type=int)
        parser.add_argument("--metric", help="Image metric",
                            required=False, default="MS")
        parser.add_argument(
            "--verbose", help="Log level (0,1,2)", default=1, type=int)

        # Since we are inside the subcommand now we skip the first two
        # arguments on the command line
        args, unknown_args = parser.parse_known_args(sys.argv[2:])
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

        inavs = io.read_data(args.input)
        info = io.read_info(args.input)

        nT = inavs.shape[0]
        nNav = inavs.shape[1]

        fixed_image = create_itk_image(
            inavs[0, 0, :, :, :], info, dtype=itk.D, max_image_value=1E3)
        if args.mask:
            mask_image = create_itk_image(io.read_data(
                args.mask), io.read_info(args.mask), dtype=itk.D)
        else:
            mask_image = None
        reg = list()
        for ij in range(0, nT):
            for ii in range(0, nNav):
                moving_image = create_itk_image(
                    inavs[ij, ii, :, :, :], info, dtype=itk.D)
                reg = versor3D_registration(fixed_image=fixed_image,
                                            moving_image=moving_image,
                                            mask_image=mask_image,
                                            sigmas=args.sigma,
                                            shrink=args.shrink,
                                            metric=args.metric,
                                            verbose=args.verbose,
                                            **more_args)
                print(reg)


def main():
    # Everything is executed by initialising this class.
    # The command line arguments will be parsed and the appropriate function will be called
    Merlin_parser()


if __name__ == '__main__':
    main()
