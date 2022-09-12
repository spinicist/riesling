Data Format
===========

RIESLING uses `HDF5 <https://www.hdfgroup.org/solutions/hdf5>`_ to store input, intermediate and output data. A command is provided to convert the ``.h5`` format to NiFTI (``.nii``).

RIESLING mandates that the non-cartesian data is stored in "traces", which could equally be called frames or traces. Here we will treat the data as being stored in `S` traces, with `N` data-points per spoke, and `C` k-space channels. In RIESLING a "frame" is a collection of traces that should be reconstructed together - e.g. from a particular echo or position in the cardiac cycle.

**Important** HDF5 uses a row-major convention, if your software is column major (RIESLING is internally) then the order of the dimensions given below should be reversed.

In contrast to BART, RIESLING uses named datasets within the ``.h5`` file. The names correspond to the steps in the reconstruction pipeline. The key ones are:
1. ``noncartesian`` - The noncartesian input k-space data
2. ``cartesian`` - K-space after the gridding operation to the Cartesian grid
3. ``channels`` - Separate channel images after Fourier Transfrom to image space
4. ``image`` - The reconstructed image
In addition, the header ``info`` and ``trajectory`` are required at all steps. Below are details of each of these.

Header
------

To be considered valid RIESLING input, the HDF5 file must contain the header information datastructure, stored as a compound data-type in a dataset with the name ``info``. We reserve the right to change these fields of the header structure between versions of RIESLING. For the canonical definition of the header, see ``src/info.hpp``. A pseudo-code version of the header is given here for clarity:

.. code-block:: c

  struct Info {
    long type;
    long matrix[3];

    long channels;
    long samples;
    long traces;

    long volumes;
    long frames;

    float tr;
    float voxel_size[3];
    float origin[3];
    float direction[3][3];
  };

* ``type`` defines the kind of acquisition. Currently two values are supported - 1 means the acquisition is fully 3D, while 2 means the acquisition is a 3D stack-of-stars or stack-of-spirals type acquisition, with cartesian phase-encoding blips for the third axis.
* ``matrix`` defines the nominal matrix size for the scan - i.e. it determines the matrix size of the final reconstructed image (unless the `--fov` option is used).
* ``channels`` defines the number of k-space channels / coil-elements.
* ``samples`` sets the number of data-points in the readout direction, i.e. how many readout points per spoke.
* ``traces`` sets the number of traces in the non-cartesian k-space acquisition.
* ``volumes`` indicates how many volumes or time-points were acquired in the acquisition.
* ``frames`` specifies how many separate frames (or echoes) were acquired per volume.

The final four fields specify the TR and image orientation as required to build a valid NIfTI output file.

* ``tr`` The repetition time. Should be specified in milliseconds as per NIfTI convention.
* ``voxel_size`` The nominal voxel-size. Should be specified in millimeters as per NIfTI/ITK convention.
* ``origin`` The physical-space location of the center of the voxel at index 0,0,0, as per ITK convention.
* ``direction`` The physical-space axes directions, as per ITK convention.

Non-cartesian Data
------------------

The non-cartesian data must be stored in a complex-valued float-precision dataset named ``noncartesian`` with dimensions ``V,S,N,C`` where V is the number of volumes. HDF5 does not have a native complex-valued datatype, hence a compound datatype with ``r`` and ``i`` members corresponding to the real and imaginary parts must be used. In contrast to other toolkits RIESLING stores the channels as the fastest-varying index, i.e. the data for each k-space point across all channels is stored contiguously.

Trajectory
----------

The trajectory should be stored as a float array in a dataset with the name ``trajectory`` with dimensions ``S,N,3``. The 3 co-ordinates correspond to the x, y & z locations within the k-space volume. For a full 3D acquisition these should be scaled such that the nominal edge of k-space in each direction is 0.5. Hence, for radial traces the k-space locations go between 0 and 0.5, and for diameter traces between -0.5 and 0.5. For a 3D stack trajectory, the z co-ordinate should be the slice/stack position.

The trajectory is assumed to repeat for each volume.

Frames
------

If the trajectory (and corresponding data) contains multiple frames, e.g. temporal points, echoes or respiratory phases, which logically form separate images, then an additional dataset should be added to the input H5 file called ``frames``. This should be a zero-based integer valued, one-dimensional array with the number of entries equal to the number of traces specified in the ``info`` structure. Each entry specifies the frame that each spoke should be allocated to.

The key difference between a frame and a volume is that all frames will have the NuFFT applied simultaneously, i.e. they will be gridded and Fourier Transformed together, whereas volumes will be reconstructed completely separately.

Cartesian k-Space
-----------------

The ``riesling grid --adj`` command will produce a complex-valued dataset named ``cartesian`` containing the gridded cartesian data for all channels. The dimensions will depend on the reconstruction settings (notably the oversampling factor).

Channel Images
--------------

The ``riesling nufft --adj`` command will produce complex-valued individual channel images in the ``channels`` dataset. The dimensions will be ```GZ,GY,GX,F,C`` where GX, GY & GZ are the grid dimensions (determined by the oversampling factor) and F is the number of frames.

Image Data
----------

The output of a reconstruction command will write a complex-valued dataset named ``image``, unless the ``--mag`` command is specified in which case the dataset will be real-valued. The dimensions will be ``V,Z,Y,X,F`` where V is the number of volumes, X, Y & Z are the matrix size as specified in ``info`` (unless the ``--fov`` argument was used), and F is the number of frames.

Density Compensation
--------------------

``riesling sdc`` pre-calculates Sample Density Correction factors. It produces a real-valued dataset ``sdc`` of dimension ``S,N``.

Meta-Information
----------------

RIESLING is capable of storing additional meta-information and passing it through the processing chain. This should be stored in an HDF5 group named ``meta``, and consist of key-value pairs where the key is the dataset name and the value is a single floating-point number.
