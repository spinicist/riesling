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

    return rx, ry, rz

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

def versor_reg_summary(registrations, reg_outs, names=None, doprint=True, show_legend=True):
    """Summarise results from one or more versor registration experiments

    Args:
        registrations (list): List of registration objects
        reg_outs (list): List of dictionaries of registration outputs
        names (list, optional): Labels for each registration. Defaults to None.
        doprint (bool, optional): Print output. Defaults to True.
        show_legend (bool, optional): Show plot legend. Defaults to True.

    Returns:
        pandas.DataFrame: Summary of registrations
    """

    df_dict = {}
    index = ['Trans X', 'Trans Y', 'Trans Z',
             'Versor X', 'Versor Y', 'Versor Z',
             'Iterations', 'Metric Value']
    if doprint:
        fig, axes = plt.subplots(ncols=3, nrows=3, figsize=(12, 8))

    if not names:
        names = ['Int %d' % x for x in range(len(registrations))]

    for (reg, reg_out, name) in zip(registrations, reg_outs, names):
        # Examine the result
        transform = reg.GetTransform()
        optimizer = reg.GetOptimizer()
        final_parameters = transform.GetParameters()

        versorX = final_parameters[0]
        versorY = final_parameters[1]
        versorZ = final_parameters[2]
        transX = final_parameters[3]
        transY = final_parameters[4]
        transZ = final_parameters[5]
        nits = optimizer.GetCurrentIteration()
        best_val = optimizer.GetValue()

        # Summarise data and store in dictionary
        reg_data = [transX, transY, transZ,
                    versorX, versorY, versorZ,
                    nits, best_val]
        df_dict[name] = reg_data

        # Creat plots
        if doprint:
            ax = axes[0, 0]
            ax.plot(reg_out['cv'], '-o')
            ax.set_ylabel('')
            ax.set_title('Optimizer Value')
            ax.grid('on')

            ax = axes[0, 1]
            ax.plot(reg_out['lrr'], '-o')
            ax.set_ylabel('')
            ax.set_title('Learning Rate Relaxation')
            ax.grid('on')

            ax = axes[0, 2]
            ax.plot(reg_out['sl'], '-o', label=name)
            ax.set_ylabel('')
            ax.set_title('Step Length')
            ax.grid('on')
            if show_legend:
                ax.legend()

            ax = axes[1, 0]
            ax.plot(reg_out['tX'], '-o')
            ax.set_ylabel('[mm]')
            ax.set_title('Translation X')
            ax.grid('on')

            ax = axes[1, 1]
            ax.plot(reg_out['tY'], '-o')
            ax.set_ylabel('[mm]')
            ax.set_title('Translation Y')
            ax.grid('on')

            ax = axes[1, 2]
            ax.plot(reg_out['tZ'], '-o')
            ax.set_ylabel('[mm]')
            ax.set_title('Translation Z')
            ax.grid('on')

            ax = axes[2, 0]
            ax.plot(reg_out['vX'], '-o')
            ax.set_xlabel('Itteration')
            ax.set_ylabel('')
            ax.set_title('Versor X')
            ax.grid('on')

            ax = axes[2, 1]
            ax.plot(reg_out['vY'], '-o')
            ax.set_xlabel('Itteration')
            ax.set_ylabel('')
            ax.set_title('Versor Y')
            ax.grid('on')

            ax = axes[2, 2]
            ax.plot(reg_out['vZ'], '-o')
            ax.set_xlabel('Itteration')
            ax.set_ylabel('')
            ax.set_title('Versor Z')
            ax.grid('on')

    if doprint:
        plt.tight_layout()
        plt.show()

    # Create Dataframe with output data
    df = pd.DataFrame(df_dict, index=index)

    # Determine if running in notebook or shell to get right print function
    env = os.environ
    program = os.path.basename(env['_'])

    if doprint:
        print(df)

    return df

def versor_watcher(reg_out, optimizer):
    """Logging for registration

    Args:
        reg_out (dict): Structure for logging registration
        optimizer (itk.RegularStepGradientDescentOptimizerv4): Optimizer object

    Returns:
        function: Logging function
    """

    logging.debug("{:s} \t {:6s} \t {:6s} \t {:6s} \t {:6s} \t {:6s} \t {:6s} \t {:6s}".format(
        'Itt', 'Value', 'vX', 'vY', 'vZ', 'tX', 'tY', 'tZ'))

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

def resample_image(registration, moving_image, fixed_image):
    """Resample image with registration parameters

    Args:
        registration (itk.ImageRegistrationMethodv4): Registration object
        moving_image (itk.Image): Moving image
        fixed_image (itk.Image): Fixed image

    Returns:
        itk.ResampleImageFilter: Resampler filter
    """
    logging.info("Resampling moving image")
    transform = registration.GetTransform()
    final_parameters = transform.GetParameters()

    TransformType = itk.VersorRigid3DTransform[itk.D]
    finalTransform = TransformType.New()
    finalTransform.SetFixedParameters(
        registration.GetOutput().Get().GetFixedParameters())
    finalTransform.SetParameters(final_parameters)

    ResampleFilterType = itk.ResampleImageFilter[type(moving_image),
                                                 type(moving_image)]
    resampler = ResampleFilterType.New()
    resampler.SetTransform(finalTransform)
    resampler.SetInput(moving_image)

    resampler.SetSize(fixed_image.GetLargestPossibleRegion().GetSize())
    resampler.SetOutputOrigin(fixed_image.GetOrigin())
    resampler.SetOutputSpacing(fixed_image.GetSpacing())
    resampler.SetOutputDirection(fixed_image.GetDirection())
    resampler.SetDefaultPixelValue(0)
    resampler.Update()

    return resampler


def get_versor_factors(registration):
    """Calculate correction factors from Versor object

    Args:
        registration (itk.ImageRegistrationMethodv4): Registration object

    Returns:
        dict: Correction factors
    """

    transform = registration.GetTransform()
    final_parameters = transform.GetParameters()

    TransformType = itk.VersorRigid3DTransform[itk.D]
    finalTransform = TransformType.New()
    finalTransform.SetFixedParameters(
        registration.GetOutput().Get().GetFixedParameters())
    finalTransform.SetParameters(final_parameters)

    matrix = itk.array_from_matrix(finalTransform.GetMatrix())
    offset = np.array(finalTransform.GetOffset())
    regParameters = registration.GetOutput().Get().GetParameters()

    corrections = {'R': matrix,
                   'vx': regParameters[0],
                   'vy': regParameters[1],
                   'vz': regParameters[2],
                   'dx': regParameters[3],
                   'dy': regParameters[4],
                   'dz': regParameters[5]
                   }

    return corrections


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
    logging.info("Number of itterations: %d" % nit)
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


def versor3D_registration(fixed_image_fname,
                         moving_image_fname,
                         moco_output_name=None,
                         fixed_output_name=None,
                         fixed_mask_fname=None,
                         reg_par_name=None,
                         iteration_log_fname=None,
                         opt_range=[np.deg2rad(1), 10],
                         init_angle=0,
                         init_axis=[0, 0, 1],
                         relax_factor=0.5,
                         winsorize=None,
                         threshold=None,
                         sigmas=[0],
                         shrink=[1],
                         metric='MS',
                         learning_rate=5,
                         convergence_window_size=10,
                         convergence_value=1E-6,
                         min_step_length=1E-6,
                         nit=250,
                         verbose=2,
                         frame_index=0):
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
        frame_index (int, optional): Index of frame to register. Defaults to 0.

    Returns:
        tuple: (registration, reg_out, reg_par_name)
    """
    # Logging setup
    log_level = {0: None, 1: logging.INFO, 2: logging.DEBUG}
    logging.basicConfig(format="[%(asctime)s] %(levelname)s: %(message)s", level=log_level[verbose],datefmt="%I:%M:%S")

    # Global settings
    PixelType = itk.D
    ImageType = itk.Image[PixelType, 3]

    # Validate inputs
    if len(sigmas) != len(shrink):
        logging.error("Sigma and Shrink arrays not the same length")
        raise ValueError("Sigma and Shrink arrays must be same length")

    frame_index = int(frame_index)
    
    logging.info(f"Reading fixed image (reference frame 0): {fixed_image_fname}")
    data_fix, spacing_fix = read_navigator_frame_h5(fixed_image_fname, vol=0)
    logging.info(f"Reading moving frame {frame_index}: {moving_image_fname}")
    data_move, spacing_move = read_navigator_frame_h5(moving_image_fname, vol=frame_index)


    # Create ITK images
    fixed_image = create_image(data_fix, spacing_fix, dtype=itk.D, max_image_value=1E3)
    moving_image = create_image(data_move, spacing_move, dtype=itk.D, max_image_value=1E3)

    # Apply Winsorize filter if requested
    if winsorize:
        logging.info("Winsorising images")
        fixed_win_filter = winsorize_image(fixed_image, winsorize[0], winsorize[1])
        moving_win_filter = winsorize_image(moving_image, winsorize[0], winsorize[1])
        fixed_image = fixed_win_filter.GetOutput()
        moving_image = moving_win_filter.GetOutput()

    # Apply thresholding if requested
    if threshold == 'otsu':
        logging.info("Calculating Otsu filter")
        filt = otsu_filter(fixed_image)
        otsu_threshold = filt.GetThreshold()
        logging.info(f"Applying thresholding at Otsu threshold of {otsu_threshold}")
        fixed_image = threshold_image(fixed_image, otsu_threshold)
        moving_image = threshold_image(moving_image, otsu_threshold)
    elif threshold is not None:
        logging.info(f"Thresholding images at {threshold}")
        fixed_image = threshold_image(fixed_image, threshold)
        moving_image = threshold_image(moving_image, threshold)

    # Setup image metric
    if metric == 'MI':
        nbins = 16
        logging.info(f"Using Mattes Mutual Information image metric with {nbins} bins")
        metric = itk.MattesMutualInformationImageToImageMetricv4[ImageType, ImageType].New()
        metric.SetNumberOfHistogramBins(nbins)
        metric.SetUseMovingImageGradientFilter(False)
        metric.SetUseFixedImageGradientFilter(False)
    else:
        logging.info("Using Mean Squares image metric")
        metric = itk.MeanSquaresImageToImageMetricv4[ImageType, ImageType].New()

    # Setup versor transform
    logging.info("Initialising Versor Rigid 3D Transform")
    TransformType = itk.VersorRigid3DTransform[PixelType]
    TransformInitializerType = itk.CenteredTransformInitializer[TransformType, ImageType, ImageType]

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

    # Setup optimizer
    optimizer = setup_optimizer(PixelType, opt_range, relax_factor, nit=int(nit),
                              learning_rate=learning_rate, 
                              convergence_window_size=int(convergence_window_size),
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
    if fixed_mask_fname:
        logging.info(f"Loading fixed mask from file: {fixed_mask_fname}")
        MaskType = itk.ImageMaskSpatialObject[3]
        mask = MaskType.New()
        data_mask_fix, spacing_mask_fix = read_image_h5(fixed_mask_fname)
        mask_img = create_image(data_mask_fix, spacing_mask_fix, dtype=itk.UC)
        mask.SetImage(mask_img)
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

    # Get and log results
    corrections = get_versor_factors(registration)
    rot_x, rot_y, rot_z = versor_to_euler(
        [corrections['vx'], corrections['vy'], corrections['vz']])

    logging.info(f"Frame {frame_index} registration results:")
    logging.info("Rotation: (%.2f, %.2f, %.2f) deg" %
                (np.rad2deg(rot_x), np.rad2deg(rot_y), np.rad2deg(rot_z)))
    logging.info("Translation: (%.2f, %.2f, %.2f) mm" %
                (corrections['dx'], corrections['dy'], corrections['dz']))

    # Generate output filenames if not provided
    if not reg_par_name:
        reg_par_name = f"reg_frame0_2_frame{frame_index}.p"

    # Save outputs
    if moco_output_name:
        resampler = resample_image(registration, moving_image, fixed_image)
        writer = itk.ImageFileWriter[ImageType].New()
        writer.SetFileName(moco_output_name)
        writer.SetInput(resampler.GetOutput())
        writer.Update()

    if fixed_output_name:
        writer = itk.ImageFileWriter[ImageType].New()
        writer.SetFileName(fixed_output_name)
        writer.SetInput(fixed_image)
        writer.Update()

    if iteration_log_fname:
        with open(iteration_log_fname, 'wb') as f:
            pickle.dump(reg_out, f)

    with open(reg_par_name, 'wb') as f:
        pickle.dump(corrections, f)

    return registration, reg_out, reg_par_name
    


def histogram_threshold_estimator(img, plot=False, nbins=200):
    """Estimate background intensity using histogram.

    Initially used to reduce streaking in background but found to make little difference.

    Args:
        img (np.array): Image
        plot (bool, optional): Plot result. Defaults to False.
        nbins (int, optional): Number of histogram bins. Defaults to 200.
    """
    def smooth(x, window_len=10, window='hanning'):
        """smooth the data using a window with requested size.

        This method is based on the convolution of a scaled window with the signal.
        The signal is prepared by introducing reflected copies of the signal 
        (with the window size) in both ends so that transient parts are minimized
        in the begining and end part of the output signal.

        input:
            x: the input signal 
            window_len: the dimension of the smoothing window; should be an odd integer
            window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
                flat window will produce a moving average smoothing.

        output:
            the smoothed signal

        example:

        t=linspace(-2,2,0.1)
        x=sin(t)+randn(len(t))*0.1
        y=smooth(x)

        see also: 

        numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
        scipy.signal.lfilter

        TODO: the window parameter could be the window itself if an array instead of a string
        NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.

        -> Obtained from the scipy cookbook at: https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
        Modified to use np instead of numpy
        """

        if x.ndim != 1:
            raise(ValueError, "smooth only accepts 1 dimension arrays.")

        if x.size < window_len:
            raise(ValueError, "Input vector needs to be bigger than window size.")

        if window_len < 3:
            return x

        s = np.r_[x[window_len-1:0:-1], x, x[-2:-window_len-1:-1]]
        # print(len(s))
        if window == 'flat':  # moving average
            w = np.ones(window_len, 'd')
        else:
            w = eval('np.'+window+'(window_len)')

        y = np.convolve(w/w.sum(), s, mode='valid')

        return y[int(window_len/2-1):-int(window_len/2)]

    y, x = np.histogram(abs(img.flatten()), bins=nbins)
    x = (x[1:]+x[:-1])/2
    y = smooth(y)

    dx = (x[1:]+x[:-1])/2
    dx2 = (dx[1:]+dx[:-1])/2
    dy = np.diff(y)
    dy2 = np.diff(smooth(dy))

    # Peak of histogram
    imax = np.argmax(y)

    # Find max of second derivative after this peak
    dy2max = np.argmax(dy2[imax:])
    thr = int(dx2[imax+dy2max])

    if plot:
        plt.figure()
        plt.plot(x, y/max(y), label='H')
        plt.plot(dx, dy/np.max(dy), label='dH/dx')
        ldy2 = plt.plot(dx2, dy2/max(abs(dy2)), label=r'$dH^2/dx^2$')
        plt.axis([0, 1500, -1, 1])

        thr = int(dx2[imax+dy2max])
        plt.plot([dx2[imax+dy2max], dx2[imax+dy2max]], [-1, 1], '--',
                 color=ldy2[0].get_color(), label='Thr=%d' % thr)
        plt.legend()
        plt.show()

    return thr

    #################################################################
    # Legacy functions
    #################################################################


def versor_resample(registration, moving_image, fixed_image):

    Dimension = 3
    PixelType = itk.D
    FixedImageType = itk.Image[PixelType, Dimension]
    MovingImageType = itk.Image[PixelType, Dimension]

    transform = registration.GetTransform()
    final_parameters = transform.GetParameters()

    TransformType = itk.VersorRigid3DTransform[itk.D]
    finalTransform = TransformType.New()
    finalTransform.SetFixedParameters(
        registration.GetOutput().Get().GetFixedParameters())
    finalTransform.SetParameters(final_parameters)

    ResampleFilterType = itk.ResampleImageFilter[MovingImageType,
                                                 FixedImageType]
    resampler = ResampleFilterType.New()
    resampler.SetTransform(finalTransform)
    resampler.SetInput(moving_image)

    resampler.SetSize(fixed_image.GetLargestPossibleRegion().GetSize())
    resampler.SetOutputOrigin(fixed_image.GetOrigin())
    resampler.SetOutputSpacing(fixed_image.GetSpacing())
    resampler.SetOutputDirection(fixed_image.GetDirection())
    resampler.SetDefaultPixelValue(0)
    resampler.Update()

    return resampler.GetOutput()