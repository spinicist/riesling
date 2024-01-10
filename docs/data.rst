Data Format
===========

RIESLING uses `HDF5 <https://www.hdfgroup.org/solutions/hdf5>`_ to store input, intermediate and output data. A command is provided to convert reconstructed images in ``.h5`` format to NiFTI (``.nii``).

Complex Numbers
---------------

All data in RIESLING is complex-valued. HDF5 does not have a native complex-valued datatype, hence a compound datatype with ``r`` and ``i`` members corresponding to the real and imaginary parts must be used.

Dimension Order
---------------

RIESLING uses *column-major / fastest-varying first convention* internally. The dimensions below are written in column-major order. However HDF5 uses row-major convention. RIESLING reverses the order of the dimensions when reading/writing files from/to HDF5. The Matlab and Python file utilities handle this reversal correctly. If constructing your own HDF5 file for use with RIESLING please take care with the order of the dimensions.

Dimensions
----------

The terminology in RIESLING is:
- "Channel" corresponds to one element in a multi-channel/multi-element receive coil. In contrast to other toolkits RIESLING stores the channels as the fastest-varying index, i.e. the data for each k-space point across all channels is stored contiguously.
- "Sample" is a single complex-valued k-space sample.
- "Trace" is the group of samples acquired during one acquisition / TR. This corresponds to a line of k-space in a Cartesian sequence, or the spoke in a radial sequence.
- "Slab" is a contiguously excited region of k-space that should be reconstructed separately.
- X, Y, Z are the image dimensions.
- "Volume" is an independent group of data that should be reconstructed separately.

RIESLING supports simultaneous reconstruction of multiple images from the same k-space data. The main useage of this is subspace reconstruction, but it also supports dynamic reconstructions with regularization of the time dimension.

In contrast to BART, RIESLING uses named datasets within the ``.h5`` file. The names correspond to the steps in the reconstruction pipeline. The key ones are:
1. ``noncartesian`` - The noncartesian input k-space data. 5D - (channel, sample, trace, slab, volumes)
2. ``cartesian`` - K-space after the gridding operation to the Cartesian grid. 6D - (channel, image, X, Y, Z, volume)
3. ``channels`` - Separate channel images after Fourier Transfrom to image space 6D - (channel, image, X, Y, Z, volume)
4. ``image`` - The reconstructed images. 5D - (image, X, Y, Z, volume)
In addition, the header ``info`` and ``trajectory`` are required at all steps. Below are details of each of these.

Header
------

To be considered valid RIESLING input, the HDF5 file must contain the header information datastructure, stored as a compound data-type in a dataset with the name ``info``. We reserve the right to change these fields of the header structure between versions of RIESLING. For the canonical definition of the header, see ``src/info.hpp``. A pseudo-code version of the header is given here for clarity:

.. code-block:: c

  struct Info {
    long matrix[3];

    float voxel_size[3];
    float origin[3];
    float direction[3][3];
    float tr;
  };

* ``matrix`` defines the nominal matrix size for the scan - i.e. it determines the matrix size of the final reconstructed image (unless the `--fov` option is used).
* ``voxel_size`` The nominal voxel-size. Should be specified in millimeters as per NIfTI/ITK convention.
* ``origin`` The physical-space location of the center of the voxel at index 0,0,0, as per ITK convention.
* ``direction`` The physical-space axes directions, as per ITK convention.
* ``tr`` The repetition time. Should be specified in milliseconds as per NIfTI convention.

Trajectory
----------

The trajectory should be stored as a float array in a dataset with the name ``trajectory`` with dimensions ``N,samples,traces``, where ``N`` is the number of co-ordinates (2 or 3). The co-ordinates correspond to the x, y (& z) locations within the k-space volume. For a full 3D acquisition these should be scaled such that the nominal edge of k-space in each direction is 0.5. Hence, for radial traces the k-space locations go between 0 and 0.5, and for diameter traces between -0.5 and 0.5.

The trajectory is assumed to repeat for each slab and volume.

Meta-Information
----------------

RIESLING is capable of storing additional meta-information and passing it through the processing chain. This should be stored in an HDF5 group named ``meta``, and consist of key-value pairs where the key is the dataset name and the value is a single floating-point number.
