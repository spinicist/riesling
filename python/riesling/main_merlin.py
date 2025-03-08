#!/usr/bin/env python3

"""
Main script to run MERLIN functions on the command line. Uses ``.h5`` files as input, assuming that the dataset ``image`` is occupied by the image data. 

The executable works like the git command with subcommands.

.. code:: bash

    usage: pymerlin <command> [<args>]

    Available commands are:
        reg         Register data
        merge       Merge registration into series
        moco        Run moco
        report      View report of data
        metric      Image metric analysis
        animation   Navigator and registration animation
        ssim        Calculate Structural Similarity Index Measure
        aes         Calculate Average Edge Strength
        nrmse       Calculate Normalised Root Mean Squared Error
        tukey       Applies Tukey filter to radial k-space data
        param       Makes a valid parameter file

To get more help for a specific command add ``-h``.

.. code:: bash

    >> pymerlin reg -h

    usage: pymerlin reg [<args>]

    MERLIN Registration

    optional arguments:
    -h, --help            show this help message and exit
    --fixed FIXED         Fixed image
    --moving MOVING       Moving image
    --reg REG             Registration parameters
    --log LOG             Registration history log
    --fixout FIXOUT       Name of fixed image output
    --moveout MOVEOUT     Name of registered moving image output
    --rad RAD             Radius of fixed mask
    --thr THR             Low image threshold
    --sigma SIGMA [SIGMA ...]
                            List of sigmas
    --shrink SHRINK [SHRINK ...]
                            Shrink factors
    --metric METRIC       Image metric
    --verbose VERBOSE     Log level (0,1,2)

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

from .dataIO import (arg_check_h5, arg_check_nii, get_merlin_fields, make_3D, parse_fname,
                     read_image_h5)
from .iq import aes, nrmse, ssim, gradient_entropy
from .moco import moco_combined, moco_single, moco_sw
from .plot import reg_animation, report_plot
from .reg import versor3D_registration, histogram_threshold_estimator
from .utils import make_tukey


class PyMerlin_parser(object):
    """
    Class to produce a subcommand arg parser like git for pyMERLIN

    Inspired by: https://chase-seibert.github.io/blog/2014/03/21/python-multilevel-argparse.html
    """

    def __init__(self):
        parser = argparse.ArgumentParser(description='MERLIN Python tools',
                                         usage='''pymerlin <command> [<args>]

    Available commands are:
        reg         Register data
        merge       Merge registration into series
        moco        Run moco
        report      View report of data
        metric      Image metric analysis
        animation   Navigator and registration animation
        ssim        Calculate Structural Similarity Index Measure
        aes         Calculate Average Edge Strength
        nrmse       Calculate Normalised Root Mean Squared Error
        tukey       Applies Tukey filter to radial k-space data
        param       Makes a valid parameter file
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
                                         usage='pymerlin reg [<args>] [versor3D_registration arguments]')

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
        #remember to remove 
        parser.add_argument('--bad-frames', type=str, help='Comma-separated list of frame indices to remove')


        # Since we are inside the subcommand now we skip the first two
        # arguments on the command line
        args, unknown_args = parser.parse_known_args(sys.argv[2:])
        main_reg(args, unknown_args)

    def merge(self):
        parser = argparse.ArgumentParser(description='Append to registration or initialise series',
                                         usage='pymerlin merge [<args>]')
        parser.add_argument("--input", nargs='+', help="Reg input. Initialize by setting input to 0",
                            required=False)
        parser.add_argument("--reg", help="Output reg object to save or append to",
                            required=True)
        parser.add_argument("--spi", help="Spokes per interleave",
                            type=int, required=True)
        parser.add_argument(
            "--verbose", help="Log level (0,1,2)", default=2, type=int)

        args = parser.parse_args(sys.argv[2:])
        main_merge(args)

    def moco(self):
        parser = argparse.ArgumentParser(
            description="Moco of complete series interleave", usage='pymerlin moco [<args>]')
        parser.add_argument("--input", help="Input H5 k-space",
                            required=True, type=arg_check_h5)
        parser.add_argument("--output", help="Output correct H5 k-space",
                            required=True, type=arg_check_h5)
        parser.add_argument(
            "--reg", help="All registration parameters in combined file", required=True, type=str, default=None)
        parser.add_argument(
            "--nseg", help="Segments per interleave for sliding window", required=False, type=int, default=None
        )
        parser.add_argument(
            '--waspi', help='Apply moco to WASPI data', action='store_true')
        # parser.add_argument(
        #     '--nlores', help='Number of WASPI spokes', type=int, default=0)
        parser.add_argument(
            "--verbose", help="Log level (0,1,2)", default=2, type=int)

        args = parser.parse_args(sys.argv[2:])
        main_moco(args)

    def thr(self):
        parser = argparse.ArgumentParser(
            description="Quick navigator background threshold estimator", usage='pymerlin thr [<args>]')
        parser.add_argument("--input", help="Input H5 navigator",
                            required=True, type=arg_check_h5)
        parser.add_argument("--nbins", help="Number of bins",
                            type=int, default=200)
        parser.add_argument("--plot", help="Show plot", action='store_true')

        args = parser.parse_args(sys.argv[2:])
        main_thr(args)

    def report(self):
        parser = argparse.ArgumentParser(
            description="Moco report", usage='pymerlin report [<args>]')
        parser.add_argument("--reg", help="Combined registration object",
                            required=True)
        parser.add_argument(
            "--out", help="Output name of figure (.png)", required=False, type=str, default='regstats.png')
        parser.add_argument(
            "--navtr", help="Navigator duration (s)", required=False, type=float)
        parser.add_argument(
            "--maxd", help="Max y-range translation", required=False, default=0)
        parser.add_argument(
            "--maxr", help="Max y-range rotation", required=False, default=0)
        parser.add_argument("--bw", action='store_true', default=False,
                            required=False, help="Plot in black and white")
        args = parser.parse_args(sys.argv[2:])
        main_report(args)

    def metric(self):
        parser = argparse.ArgumentParser(
            description="Image metrics", usage="pymerlin metric [<args>]")
        parser.add_argument("--input", help="Image input", required=True)

        args = parser.parse_args(sys.argv[2:])
        main_metric(args)

    def animation(self):
        parser = argparse.ArgumentParser(
            description="Make animation from navigator and reg results", usage='pymerlin gif [<args>]')
        parser.add_argument("--reg", help="Combined registration object",
                            required=True)
        parser.add_argument("--nav", help="Navigator folder", required=True)
        parser.add_argument("--out", help="Output gif name",
                            required=False, default="reg_animation.gif")
        parser.add_argument(
            "-x", help="Slice x, def middle)", required=False, default=None, type=int)
        parser.add_argument(
            "-y", help="Slice y, def middle)", required=False, default=None, type=int)
        parser.add_argument(
            "-z", help="Slice z, def middle)", required=False, default=None, type=int)
        parser.add_argument("--rot", help="Rotations to slices",
                            required=False, default=0, type=int)
        parser.add_argument(
            "--navtr", help="Navigator duration (s)", required=False, type=float)
        parser.add_argument("--t0", help="Time offset",
                            required=False, default=0)
        parser.add_argument(
            "--maxd", help="Max y-range translation", required=False, default=None, type=float)
        parser.add_argument(
            "--maxr", help="Max y-range rotation", required=False, default=None, type=float)
        parser.add_argument("--vmax", help="Max display rnage normalised to max image intensity",
                            required=False, default=1, type=float)

        args = parser.parse_args(sys.argv[2:])
        main_animation(args)

    def ssim(self):
        parser = argparse.ArgumentParser(
            description="Calculate Structural Similarity Index Measure (SSIM)", usage="pymerlin ssim [<args>]")
        parser.add_argument('img1', type=str,
                            help='Reference image')
        parser.add_argument('img2', type=str,
                            help='Comparison image')
        parser.add_argument('--kw', required=False, default=11,
                            type=int, help='Kernel width')
        parser.add_argument('--sigma', required=False, default=1.5,
                            type=float, help='Sigma for Gaussian kernel')
        parser.add_argument('--mask', required=False,
                            default=None, help='Brain mask')
        parser.add_argument('--out', required=False,
                            default='ssim.nii.gz', type=str, help='Output filename')

        args = parser.parse_args(sys.argv[2:])
        main_ssim(args)

    def aes(self):
        parser = argparse.ArgumentParser(
            description="Calculate the Average Edge Strength (AES)", usage="pymerlin aes [<args>]")
        parser.add_argument('img', type=str, help='Input image')
        parser.add_argument('--mask', type=str,
                            help='Brain mask', required=False, default=None)
        parser.add_argument('--canny', type=str,
                            help='Canny edge mask', required=False, default=None)
        parser.add_argument(
            '--sigma', type=float, help='Canny edge filter sigma', required=False, default=2)
        parser.add_argument('--out', type=str,
                            help='Output folder', required=False, default='.')

        args = parser.parse_args(sys.argv[2:])
        main_aes(args)

    def nrmse(self):
        parser = argparse.ArgumentParser(
            description="Calculate the Normalised Root Mean Squared Error (NRMSE)", usage="pymerlin nrmse [<args>]")
        parser.add_argument("--ref", type=str,
                            help="Reference image", required=True)
        parser.add_argument("--comp", type=str,
                            help="Comparison image", required=True)
        parser.add_argument("--mask", type=str, help="Mask", required=True)
        parser.add_argument("--out", type=str,
                            help="Output folder", required=False, default='.')

        args = parser.parse_args(sys.argv[2:])
        main_nrmse(args)

    def tukey(self):
        parser = argparse.ArgumentParser(
            description="Applies a tukey filter to radial k-space data",
            usage="pymerlin tukey [<args>]")
        parser.add_argument("--input", type=str,
                            help="Input data", required=True)
        parser.add_argument("--output", type=str,
                            help="Output data", required=True)
        parser.add_argument("--alpha", required=False,
                            type=float, default=0.5, help="Filter width")

        args = parser.parse_args(sys.argv[2:])
        main_tukey(args)

    def param(self):
        parser = argparse.ArgumentParser(
            description='Generate a valid parameter file for run_merlin_sw',
            usage='pymerlin param par_file [<args>]')
        parser.add_argument('parfile', type=str, help='Output parameter file')
        parser.add_argument('--nspokes', type=int,
                            help='Number of spokes', required=True)
        # parser.add_argument('--nlores', type=int,
        #                     help='Number of lores spokes', required=True)
        parser.add_argument('--spoke_step', type=int,
                            help='Spoke step for sliding window', required=True)
        parser.add_argument('--make_brain_mask',
                            action='store_true', default=False)
        parser.add_argument('--brain_mask_file', type=str,
                            default=None, help='Brain mask')
        parser.add_argument('--sense_maps', type=str,
                            help='Input sense maps', default=None)
        parser.add_argument('--cg_its', type=int,
                            help='Number of cgSENSE iterations', default=4)
        parser.add_argument(
            '--ds', type=int, help='Downsampling factor for navigator recon', default=3)
        parser.add_argument('--fov', type=int,
                            help='FOV for navigator recon', default=240)
        parser.add_argument('--overwrite_files',
                            action='store_true', help='Overwrite files')
        parser.add_argument('--riesling_verbosity',
                            type=int, default=0, help='Riesling verbose output')
        parser.add_argument('--ref_nav_num', type=int,
                            help='Navigator reference', default=0)
        parser.add_argument('--metric', type=str,
                            help='Registration metric', default='MS')
        parser.add_argument('--batch_itk', type=int,
                            help='Batch size for ITK', default=4)
        parser.add_argument('--batch_riesling', type=int,
                            help='Batch size for riesling', default=4)
        parser.add_argument('--threads_itk', type=int,
                            help='Number of threads ITK', default=2)
        parser.add_argument('--threads_riesling', type=int,
                            help='Number of riesling threads', default=2)

        args = parser.parse_args(sys.argv[2:])
        main_param(args)

    def get_args(self):
        return self.outargs


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


def main_thr(args):
    img, spacing = read_image_h5(args.input)
    thr = histogram_threshold_estimator(img, args.plot, args.nbins)
    print(thr)


def main_report(args):
    combreg = pickle.load(open(args.reg, 'rb'))

    report_plot(combreg, args.maxd, args.maxr, args.navtr, args.bw)

    # Check filename
    out_name = args.out
    fname, ext = os.path.splitext(out_name)
    if ext != '.png':
        print("Warning: output extension is not .png")
        out_name = fname + '.png'
        print("Setting output name to: {}".format(out_name))

    plt.savefig(out_name, dpi=300)
    plt.show()


def main_moco(args):
    log_level = {0: None, 1: logging.INFO, 2: logging.DEBUG}
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s: %(message)s", level=log_level[args.verbose], datefmt="%I:%M:%S")

    reg_list = pickle.load(open(args.reg, 'rb'))

    if isinstance(reg_list, dict):
        moco_single(args.input, args.output, reg_list)
    elif args.nseg:
        moco_sw(args.input, args.output, reg_list, args.nseg)
    else:
        moco_combined(args.input, args.output, reg_list)

def main_merge(args):
    log_level = {0: None, 1: logging.INFO, 2: logging.DEBUG}
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s: %(message)s", level=log_level[args.verbose], datefmt="%I:%M:%S")
    if not args.input:
        logging.info("Initializing new reg structure")

        if os.path.exists(args.reg):
            logging.warning("%s already exists" % args.reg)
            raise Exception("Cannot overwrite file")

        D = {'R': np.eye(3),
             'vx': 0,
             'vy': 0,
             'vz': 0,
             'dx': 0,
             'dy': 0,
             'dz': 0,
             'spi': args.spi}

        L = []
        L.append(D)
        pickle.dump(L, open(args.reg, 'wb'))
        logging.info("New reg structure saved to {}".format(args.reg))
        logging.info(f"Initializing reg structure with spi={args.spi}")
    else:
        logging.info("Opening {}".format(args.reg))
        dlist = pickle.load(open(args.reg, 'rb'))

        for input in args.input:
            logging.info("Opening {}".format(input))
            in_dict = pickle.load(open(input, 'rb'))
            in_dict['spi'] = args.spi
            dlist.append(in_dict)

            nreg = len(dlist)

            logging.info("Adding as reg object number {}".format(nreg))

        logging.info("Writing combined reg object back to {}".format(args.reg))
        pickle.dump(dlist, open(args.reg, 'wb'))


def main_metric(args):
    img = nib.load(args.input).get_fdata()
    GE = gradient_entropy(img)

    print(GE)

def main_animation(args):
    nav_dir = args.nav
    files = os.listdir(nav_dir)
    string_match = '-nav0.'
    fbase = next(f.split(string_match) for f in files if string_match in f)
    
    file_tmpl = os.path.join(nav_dir, fbase[0] + '-nav%d.h5')
    num_files = len([f for f in files if 'moving_phantom-nav' in f])
    
    images = None
    print("Reading navigator images")
    for i in range(num_files):
        img, _ = read_image_h5(file_tmpl % i, vol=0)
        # img = img[0,:128,:,:,0]
        img = img[0,:,:,:,0]
        
        if images is None:
            images = np.zeros((85, 85, 85, num_files))
        
        images[:, :, :, i] = abs(img)
        
    reg_animation(args.reg, images, out_name=args.out,
                 tnav=args.navtr, t0=0, max_d=args.maxd, max_r=args.maxr, vmax=args.vmax, slice_pos=[args.x, args.y, args.z], nrot=args.rot)


def main_ssim(args):
    nii1 = nib.load(args.img1)
    image1 = nii1.get_fdata()
    image2 = nib.load(args.img2).get_fdata()

    if len(image1.shape) > 3:
        image1 = image1[..., 0]
    if len(image2.shape) > 3:
        image2 = image2[..., 0]

    if args.mask:
        mask = nib.load(args.mask).get_fdata()
        image1 *= mask
        image2 *= mask

    mssim, S = ssim(image1, image2, kw=args.kw, sigma=args.sigma)

    if args.mask:
        S *= mask

    mssim = np.mean(S[mask == 1])

    ssim_nii = nib.Nifti1Image(S, nii1.affine)
    nib.save(ssim_nii, args.out)
    print('MSSIM: {}'.format(mssim))
    print('Saving SSIM to: {}'.format(args.out))


def main_aes(args):

    nii = nib.load(args.img)

    img = make_3D(nib.load(args.img).get_fdata())

    if args.mask:
        mask = make_3D(nib.load(args.mask).get_fdata())
    else:
        mask = np.ones_like(img)

    if args.canny:
        canny = make_3D(nib.load(args.canny).get_fdata())
    else:
        canny = None

    img_aes, img_edges, canny_edges = aes(
        img, mask=mask, canny_edges=canny, canny_sigma=args.sigma)

    bname = parse_fname(args.img)

    edges_nii = nib.Nifti1Image(img_edges, nii.affine)
    nib.save(edges_nii, "{}/{}_edges.nii.gz".format(args.out, bname))

    if not args.canny:
        canny_nii = nib.Nifti1Image(canny_edges, nii.affine)
        nib.save(canny_nii, "{}/{}_canny.nii.gz".format(args.out, bname))

    print("Average edge strength")
    print("Image: {}".format(args.img))
    if args.canny:
        print("Canny edges: {}".format(args.canny))
    else:
        print("Canny edges calculated from image")

    print("AES:{}".format(img_aes))


def main_nrmse(args):

    ref_nii = nib.load(args.ref)

    ref_img = make_3D(ref_nii.get_fdata())
    comp_img = make_3D(nib.load(args.comp).get_fdata())
    mask = make_3D(nib.load(args.mask).get_fdata())

    # Normalise images to avoid difference in global scaling
    ref_img /= np.quantile(ref_img[mask == 1], 0.99)
    comp_img /= np.quantile(comp_img[mask == 1], 0.99)

    img_nrmse = nrmse(ref_img, comp_img, mask)

    print("Normalised Root Mean Squared Error (NRMSE)")
    print("Ref Image: {}".format(args.ref))
    print("Comparison Image: {}".format(args.comp))
    print("Mask: {}".format(args.mask))
    print("NRMSE: {}".format(img_nrmse))

    diff_img = ref_img - comp_img

    diff_nii = nib.Nifti1Image(diff_img, ref_nii.affine)

    nib.save(diff_nii, "{}/{}_diff.nii.gz".format(args.out,
             parse_fname(args.comp)))


def main_tukey(args):
    print("Applying tukey filter to {}".format(args.input))
    h5 = h5py.File(args.input, 'r')
    ks = h5['data'][0, ...]
    info = h5['info'][:]
    traj = h5['trajectory'][:]
    h5.close()

    nspokes, npoints, ndim = traj.shape
    filt = make_tukey(nspokes, a=args.alpha)
    ks_filt = np.transpose(np.transpose(ks, [1, 2, 0])*filt, [2, 0, 1])

    f_dest = h5py.File(args.output, 'w')
    f_dest.create_dataset("info", data=info)

    f_dest.create_dataset("trajectory", data=traj,
                          chunks=np.shape(traj), compression='gzip')

    f_dest.create_dataset("noncartesian", dtype='c8', data=ks_filt[np.newaxis, ...],
                          chunks=np.shape(ks_filt[np.newaxis, ...]), compression='gzip')

    f_dest.close()
    print("Saved data to {}".format(args.output))


def main_param(args):
    valid_args = get_merlin_fields()
    with open(args.parfile, 'w') as f:
        for key in valid_args:
            val = eval(f'args.{key}')
            print(f"{key}={val}")
            if type(val) == bool:
                val = int(val)
            if val == None:
                val = ''
            f.write(f'{key}={val}\n')

    print(f"Wrote parameters to: {args.parfile}")


def main():
    # Everything is executed by initialising this class.
    # The command line arguments will be parsed and the appropriate function will be called
    PyMerlin_parser()


if __name__ == '__main__':
    main()