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
import nibabel as nib
import numpy as np

from . import io
from .merlin import versor3D_registration, extract_nav

REG_FIELDS = ['R', 'delta']
REG_FORMAT = [('<f4', (3, 3)), ('<f4', (3,))]
REG_DTYPE = np.dtype({'names': REG_FIELDS, 'formats': REG_FORMAT})


class Merlin_parser(object):
    """
    Class to produce a subcommand arg parser like git for MERLIN

    Inspired by: https://chase-seibert.github.io/blog/2014/03/21/python-multilevel-argparse.html
    """

    def __init__(self):
        parser = argparse.ArgumentParser(description='MERLIN Registration',
                                         usage='merlin [<args>] [versor3D_registration arguments]')

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
            "--winsorize", help="Normalize image intensities before registration", required=False, action='store_true')
        parser.add_argument(
            "--verbose", help="Log level (0,1,2)", default=1, type=int)
        args = parser.parse_args()

        PixelType = itk.D
        inavs = itk.imread(args.input, PixelType)
        fixed_image = extract_nav(inavs, 0)
        if args.mask:
            mask_image = itk.imread(args.mask, itk.UC)
        else:
            mask_image = None
        with h5py.File(args.output, 'w') as ofile:
            for ii in range(inavs.shape[0]):
                moving_image = extract_nav(inavs, ii)
                reg = versor3D_registration(fixed_image=fixed_image,
                                            moving_image=moving_image,
                                            mask_image=mask_image,
                                            sigmas=args.sigma,
                                            shrink=args.shrink,
                                            metric=args.metric,
                                            verbose=args.verbose,
                                            winsorize=args.winsorize)

                ofile.create_dataset(f'{ii:03d}', data=np.array(
                    [tuple([reg[f] for f in REG_FIELDS])], dtype=REG_DTYPE))


def main():
    # Everything is executed by initialising this class.
    # The command line arguments will be parsed and the appropriate function will be called
    Merlin_parser()


if __name__ == '__main__':
    main()
