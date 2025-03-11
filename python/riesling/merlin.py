# -*- coding: utf-8 -*-
"""
Contains the core registration components for MERLIN. The framework builds on ITK and is heavily inspired by ANTs.
"""

import logging
import os
import pickle

import itk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import io


def extract_nav(navs, ii):
    PixelType = itk.D
    ImageType3 = itk.Image[PixelType, 3]
    ImageType4 = itk.Image[PixelType, 4]
    sz = navs.GetLargestPossibleRegion().GetSize()
    region = itk.ImageRegion[4]()
    region.SetIndex([0, 0, 0, ii])
    region.SetSize([sz[0], sz[1], sz[2], 0])
    extract = itk.ExtractImageFilter[ImageType4, ImageType3].New()
    extract.SetDirectionCollapseToSubmatrix()
    extract.SetExtractionRegion(region)
    extract.SetInput(navs)
    extract.Update()
    return extract.GetOutput()


def versor_to_euler(versor):
    """
    Calculates the intrinsic euler angles from a 3 or 4 element versor

    Args:
        versor (array): 3 or 4 element versor

    Returns:
        array: rotation angles (rx, ry, rz)
    """

    if len(versor) == 4:
        q0, q1, q2, q3 = versor

    elif len(versor) == 3:
        q1, q2, q3 = versor
        q0 = np.sqrt(1 - q1**2 - q2**2 - q3**2)
    else:
        return TypeError("Versor must be of lenfth 3 or 4")

    rz = np.arctan2(2*(q0*q1+q2*q3), (1-2*(q1**2+q2**2)))
    ry = np.arcsin(2*(q0*q2 - q3*q1))
    rx = np.arctan2(2*(q0*q3+q1*q2), 1-2*(q2**2+q3**2))

    return np.rad2deg(np.array([rx, ry, rz]))


def otsu_filter(image):
    """Applies an Otsu filter

    Args:
        image (itk.Image): Input image

    Returns:
        itk.OtsuThresholdImageFilter: Filter
    """

    print("Performing Otsu thresholding")
    OtsuOutImageType = itk.Image[itk.UC, 3]
    filt = itk.OtsuThresholdImageFilter[type(image),
                                        OtsuOutImageType].New()
    filt.SetInput(image)
    filt.Update()

    return filt


def versor_watcher(reg_out, optimizer):
    """Logging for registration

    Args:
        reg_out (dict): Structure for logging registration
        optimizer (itk.RegularStepGradientDescentOptimizerv4): Optimizer object

    Returns:
        function: Logging function
    """

    logging.debug("{:s} \t {:6s} \t {:6s} \t {:6s} \t {:6s} \t {:6s} \t {:6s} \t {:6s}".format(
        'It', 'Value', 'vX', 'vY', 'vZ', 'tX', 'tY', 'tZ'))

    def opt_watcher():
        cv = optimizer.GetValue()
        cpos = np.array(optimizer.GetCurrentPosition())
        cit = optimizer.GetCurrentIteration()
        lrr = optimizer.GetCurrentLearningRateRelaxation()
        sl = optimizer.GetCurrentStepLength()

        # Store logged values
        reg_out['cv'].append(cv)
        reg_out['vX'].append(cpos[0])
        reg_out['vY'].append(cpos[1])
        reg_out['vZ'].append(cpos[2])
        reg_out['tX'].append(cpos[3])
        reg_out['tY'].append(cpos[4])
        reg_out['tZ'].append(cpos[5])
        reg_out['sl'].append(sl)
        reg_out['lrr'].append(lrr)

        logging.debug("{:d} \t {:6.5f} \t {:6.3f} \t {:6.3f} \t {:6.3f} \t {:6.3f} \t {:6.3f} \t {:6.3f}".format(
            cit, cv, cpos[0], cpos[1], cpos[2], cpos[3], cpos[4], cpos[5]))

    return opt_watcher


def winsorize_image(image, p_low, p_high):
    """Applies winsorize filter to image

    Args:
        image (itk.Image): Input image
        p_low (float): Lower percentile
        p_high (float): Upper percentile

    Returns:
        itk.ThresholdImageFilter: Threshold filter
    """

    Dimension = 3
    PixelType = itk.template(image)[1][0]
    ImageType = itk.Image[PixelType, Dimension]

    # Histogram
    nbins = 1000  # Allows 0.001 precision
    hist_filt = itk.ImageToHistogramFilter[ImageType].New()
    hist_filt.SetInput(image)
    hist_filt.SetAutoMinimumMaximum(True)
    hist_filt.SetHistogramSize([nbins])
    hist_filt.Update()
    hist = hist_filt.GetOutput()

    low_lim = hist.Quantile(0, p_low)
    high_lim = hist.Quantile(0, p_high)

    filt = itk.ThresholdImageFilter[ImageType].New()
    filt.SetInput(image)
    filt.ThresholdBelow(low_lim)
    filt.ThresholdAbove(high_lim)
    filt.ThresholdOutside(low_lim, high_lim)

    return filt


def threshold_image(image, low_lim):
    """Threshold image at given value

    Args:
        image (itk.Image): Input image
        low_lim (float): Lower threshold

    Returns:
        itk.Image: Thresholded image
    """

    Dimension = 3
    PixelType = itk.template(image)[1][0]
    ImageType = itk.Image[PixelType, Dimension]

    thresh_filt = itk.ThresholdImageFilter[ImageType].New()
    thresh_filt.ThresholdBelow(float(low_lim))
    thresh_filt.SetOutsideValue(0)
    thresh_filt.SetInput(image)
    thresh_filt.Update()

    return thresh_filt.GetOutput()


def get_R_delta(transform):
    """Calculate correction factors from Versor object

    Args:
        transform object from output of optimizer

    Returns:
        dict: Correction factors
    """
    tfm2 = itk.VersorRigid3DTransform[itk.D].New()
    # tfm2.SetFixedParameters(transform.GetFixedParameters())
    tfm2.SetParameters(transform.GetParameters())
    matrix = itk.array_from_matrix(tfm2.GetMatrix())
    offset = np.array(tfm2.GetOffset())

    return {'R': matrix, 'delta': offset}


def setup_optimizer(PixelType, opt_range, relax_factor, nit=250, learning_rate=0.1, convergence_window_size=10, convergence_value=1E-6, min_step_length=1E-4):
    """Setup optimizer object

    Args:
        PixelType (itkCType): ITK pixel type
        opt_range (list): Range for optimizer
        relax_factor (float): Relaxation factor
        nit (int, optional): Number of iterations. Defaults to 250.
        learning_rate (float, optional): Optimizer learning rate. Defaults to 0.1.
        convergence_window_size (int, optional): Number of points to use in evaluating convergence. Defaults to 10.
        convergence_value ([type], optional): Value at which convergence is reached. Defaults to 1E-6.

    Returns:
        itk.RegularStepGradientDescentOptimizerv4: Optimizer object
    """

    logging.info("Initialising Regular Step Gradient Descent Optimizer")
    optimizer = itk.RegularStepGradientDescentOptimizerv4[PixelType].New()
    OptimizerScalesType = itk.OptimizerParameters[PixelType]
    # optimizerScales = OptimizerScalesType(
    #     initialTransform.GetNumberOfParameters())
    optimizerScales = OptimizerScalesType(6)

    # Set scales <- Not sure about this part
    rotationScale = 1.0/np.deg2rad(opt_range[0])
    translationScale = 1.0/opt_range[1]
    optimizerScales[0] = rotationScale
    optimizerScales[1] = rotationScale
    optimizerScales[2] = rotationScale
    optimizerScales[3] = translationScale
    optimizerScales[4] = translationScale
    optimizerScales[5] = translationScale
    optimizer.SetScales(optimizerScales)

    logging.info("Setting up optimizer")
    logging.info("Rot/Trans scales: {}/{}".format(opt_range[0], opt_range[1]))
    logging.info("Number of iterations: %d" % nit)
    logging.info("Learning rate: %.2f" % learning_rate)
    logging.info("Relaxation factor: %.2f" % relax_factor)
    logging.info("Convergence window size: %d" % convergence_window_size)
    logging.info("Convergence value: %f" % convergence_value)

    optimizer.SetNumberOfIterations(nit)
    optimizer.SetLearningRate(learning_rate)          # Default in ANTs
    optimizer.SetRelaxationFactor(relax_factor)
    optimizer.SetConvergenceWindowSize(convergence_window_size)
    optimizer.SetMinimumConvergenceValue(convergence_value)
    optimizer.SetMinimumStepLength(min_step_length)

    return optimizer


def versor3D_registration(fixed_image,
                          moving_image,
                          mask_image=None,
                          opt_range=[0.1, 10],
                          init=None,
                          relax_factor=0.5,
                          winsorize=False,
                          threshold=None,
                          sigmas=[0],
                          shrink=[1],
                          metric='MS',
                          learning_rate=5,
                          convergence_window_size=10,
                          convergence_value=1E-6,
                          min_step_length=1E-6,
                          nit=250,
                          verbose=2):
    """Multi-scale rigid body registration

    ITK registration framework inspired by ANTs which performs a multi-scale 3D versor registration 
    between two 3D volumes from the same navigator file. Uses frame 0 as reference and registers 
    specified frame against it.

    Default values works well. Mask for the fixed image is highly recommended for ZTE data with 
    head rest pads visible.

    Note that the outputs of the registration are versor and translation vectors. The versor is 
    the vector part of a unit normalised quaternion. To get the equivalent euler angles use 
    pymerlin.utils.versor_to_euler.

    Args:
        fixed_image_fname (str): Navigator file (.h5)
        moving_image_fname (str): Same navigator file (.h5) 
        moco_output_name (str, optional): Output moco image as nifti. Defaults to None.
        fixed_output_name (str, optional): Output fixed image as nifti. Defaults to None.
        fixed_mask_fname (str, optional): Mask for fixed image. Defaults to None.
        reg_par_name (str, optional): Name of output parameter file. Defaults to None.
        iteration_log_fname (str, optional): Name for output log file. Defaults to None.
        opt_range (list, optional): Expected range of motion [deg,mm]. Defaults to [1 rad, 10 mm].
        init_angle (float, optional): Initial angle for registration. Defaults to 0.
        init_axis (array, optional): Direction of initial rotation for registration. Defaults to [0,0,1].
        relax_factor (float, optional): Relaxation factor for optimizer. Defaults to 0.5.
        winsorize (list, optional): Limits for winsorize filter. Defaults to None.
        threshold (float, optional): Lower value for threshold filter. Defaults to None.
        sigmas (list, optional): Smoothing sigmas for multi-scale registration. Defaults to [0].
        shrink (list, optional): Shrink factors for multi-scale registration. Defaults to [1].
        metric (str, optional): Image metric for registration (MI/MS). Defaults to 'MS'.
        learning_rate (float, optional): Initial step length. Defaults to 5.
        convergence_window_size (int, optional): Length of window to calculate convergence value. Defaults to 10.
        convergence_value (float, optional): Convergence value to terminate registration. Defaults to 1E-6.
        min_step_length (float, optional): Minimum step length. Defaults to 1E-6.
        nit (int, optional): Maximum number of iterations per scale. Defaults to 250.
        verbose (int, optional): Level of debugging (0/1/2). Defaults to 2.

    Returns:
        tuple: (registration, reg_out, reg_par_name)
    """
    # Logging setup
    log_level = {0: None, 1: logging.INFO, 2: logging.DEBUG}
    logging.basicConfig(format="[%(asctime)s] %(levelname)s: %(message)s",
                        level=log_level[verbose], datefmt="%I:%M:%S")

    # Global settings
    PixelType = itk.D
    ImageType = itk.Image[PixelType, 3]

    itk.VersorRigid3DTransform.GetTypes()

    # Validate inputs
    if len(sigmas) != len(shrink):
        logging.error("Sigma and Shrink arrays not the same length")
        raise ValueError("Sigma and Shrink arrays must be same length")

    # Apply Winsorize filter if requested
    if winsorize:
        logging.info("Winsorising images")
        fixed_win_filter = winsorize_image(fixed_image, 0.05, 0.95)
        moving_win_filter = winsorize_image(moving_image, 0.05, 0.95)
        fixed_image = fixed_win_filter.GetOutput()
        moving_image = moving_win_filter.GetOutput()

    # Apply thresholding if requested
    if threshold == 'otsu':
        logging.info("Calculating Otsu filter")
        filt = otsu_filter(fixed_image)
        otsu_threshold = filt.GetThreshold()
        logging.info(
            f"Applying thresholding at Otsu threshold of {otsu_threshold}")
        fixed_image = threshold_image(fixed_image, otsu_threshold)
        moving_image = threshold_image(moving_image, otsu_threshold)
    elif threshold is not None:
        logging.info(f"Thresholding images at {threshold}")
        fixed_image = threshold_image(fixed_image, threshold)
        moving_image = threshold_image(moving_image, threshold)

    # Setup image metric
    if metric == 'MI':
        nbins = 16
        logging.info(
            f"Using Mattes Mutual Information image metric with {nbins} bins")
        metric = itk.MattesMutualInformationImageToImageMetricv4[ImageType, ImageType].New(
        )
        metric.SetNumberOfHistogramBins(nbins)
        metric.SetUseMovingImageGradientFilter(False)
        metric.SetUseFixedImageGradientFilter(False)
    else:
        logging.info("Using Mean Squares image metric")
        metric = itk.MeanSquaresImageToImageMetricv4[ImageType, ImageType].New(
        )

    # Setup versor transform
    logging.info("Initialising Versor Rigid 3D Transform")
    TransformType = itk.VersorRigid3DTransform[PixelType]
    if init is None:
        init_angle=0
        init_axis=[0, 0, 1]
        TransformInitializerType = itk.CenteredTransformInitializer[TransformType,
                                                                    ImageType, ImageType]

        initialTransform = TransformType.New()
        initializer = TransformInitializerType.New()
        initializer.SetTransform(initialTransform)
        initializer.SetFixedImage(fixed_image)
        initializer.SetMovingImage(moving_image)
        initializer.GeometryOn()
        initializer.InitializeTransform()

        VersorType = itk.Versor[itk.D]
        VectorType = itk.Vector[itk.D, 3]
        rotation = VersorType()
        axis = VectorType()
        axis[0] = init_axis[0]
        axis[1] = init_axis[1]
        axis[2] = init_axis[2]
        angle = init_angle
        rotation.Set(axis, angle)
        initialTransform.SetRotation(rotation)
    else:
        initialTransform = init

    # Setup optimizer
    optimizer = setup_optimizer(PixelType, opt_range, relax_factor, nit=int(nit),
                                learning_rate=learning_rate,
                                convergence_window_size=int(
                                    convergence_window_size),
                                convergence_value=convergence_value,
                                min_step_length=min_step_length)

    # Setup registration
    registration = itk.ImageRegistrationMethodv4[ImageType, ImageType].New()
    registration.SetMetric(metric)
    registration.SetOptimizer(optimizer)
    registration.SetFixedImage(fixed_image)
    registration.SetMovingImage(moving_image)
    registration.SetInitialTransform(initialTransform)

    # Multi-scale setup
    logging.info(f"Smoothing sigmas: {sigmas}")
    logging.info(f"Shrink factors: {shrink}")
    numberOfLevels = len(sigmas)
    shrinkFactorsPerLevel = itk.Array[itk.F](numberOfLevels)
    smoothingSigmasPerLevel = itk.Array[itk.F](numberOfLevels)

    for i in range(numberOfLevels):
        shrinkFactorsPerLevel[i] = shrink[i]
        smoothingSigmasPerLevel[i] = sigmas[i]

    registration.SetNumberOfLevels(numberOfLevels)
    registration.SetSmoothingSigmasPerLevel(smoothingSigmasPerLevel)
    registration.SetShrinkFactorsPerLevel(shrinkFactorsPerLevel)

    # Apply mask if provided
    if mask_image:
        MaskType = itk.ImageMaskSpatialObject[3]
        mask = MaskType.New()
        mask.SetImage(mask_image)
        mask.Update()
        metric.SetFixedImageMask(mask)

    # Setup registration monitoring
    reg_out = {'cv': [], 'tX': [], 'tY': [], 'tZ': [],
               'vX': [], 'vY': [], 'vZ': [], 'sl': [], 'lrr': []}

    logging.info("Running Registration")
    wf = versor_watcher(reg_out, optimizer)
    optimizer.AddObserver(itk.IterationEvent(), wf)

    # Run registration
    registration.Update()
    return registration.GetOutput().Get()
