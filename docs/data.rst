Data Format
===========

RIESLING uses `HDF5 <https://www.hdfgroup.org/solutions/hdf5>`_ to store input, intermediate and output data. A command is provided to convert reconstructed images in ``.h5`` format to NiFTI (``.nii``). The majority of RIESLING tools expect 32-bit floating-point complex data written into a dataset named ``data`` inside the .h5 file.

Complex Numbers
---------------

All data in RIESLING is complex-valued. HDF5 did not have a native complex-valued datatype until version 1.14, hence a compound datatype with ``r`` and ``i`` members corresponding to the real and imaginary parts must be used. A future version of RIESLING will use the native HDF5 complex data-type.

Dimension Order
---------------

RIESLING uses *column-major / fastest-varying first convention* internally. The dimensions below are written in column-major order. However HDF5 uses row-major convention. RIESLING reverses the order of the dimensions when reading/writing files from/to HDF5. The Matlab and Python file utilities handle this reversal correctly. If constructing your own HDF5 file for use with RIESLING please take care with the order of the dimensions.

Dimensions
----------

Previous versions of RIESLING used name datasets within the .h5 file, now the expected dataset name is simply ``data``. However, the RIESLING commands require input data to have the correct order (number of dimensions) and will output data with a fixed number of dimensions. We reserve the right to change the number of dimensions in future versions. RIESLING will output the dimension names using NetCDF format, these names are not required on input datasets. Below are the list of current data orders and names:

1. Noncartesian k-space data. 5D - (channel, sample, trace, slab, time)
2. Cartesian k-space (after gridding). 6D - (i, j, k, b, channel, time)
3. Reconstructed images. 5D - (i, j, k, b, time)
4. SENSE maps. 5D - (i, j, k, b, channel)

In the above, ``b`` refers to the index for a basis vector (subspace or time-resolved reconstruction), while the ``time`` dimension refers to discrete timepoints which are reconstructed individually (e.g. fMRI type acquisition). A ``trace`` refers to a single spoke, spiral, or line of k-space, ``sample`` refers to individual sample/data points with a ``trace``.

Geometry
--------

To be considered valid RIESLING input, the HDF5 file must contain the image geometry information, stored as a compound data-type in a dataset with the name ``info``. We reserve the right to change these fields of the header structure between versions of RIESLING. For the canonical definition of the header, see ``src/info.hpp``. A pseudo-code version of the header is given here for clarity:

.. code-block:: c

  struct Info {
    float voxel_size[3];
    float origin[3];
    float direction[3][3];
    float tr;
  };

* ``voxel_size`` The nominal voxel-size. Various default values in RIESLING are specified in mm.
* ``origin`` The physical-space location of the center of the voxel at index 0,0,0.
* ``direction`` The physical-space axes directions.
* ``tr`` The repetition time. Should be specified in milliseconds as per NIfTI convention.

Trajectory
----------

The trajectory should be stored as a 3D float array in a dataset with the name ``trajectory`` with dimensions ``N,samples,traces``, where ``N`` is the number of co-ordinates (2 or 3). The co-ordinates correspond to the x, y (& z) locations within the k-space volume. For a full 3D acquisition these should be scaled such that the nominal edge of k-space in each direction is M/2, where M is the nominal matrix size in that dimension. Hence, for radial traces the k-space locations go between 0 and M/2, and for diameter traces between -M/2 and M/2. This is the same as the BART trajectory scaling. Note that before version 1, RIESLING scaled the trajectory between -0.5 and 0.5.

If no other information is provided, RIESLING will calculate the matrix size for reconstruction directly from the trajectory (by finding the maximum co-ordinate in each dimension and multiplying by two). In various situations this may not be the desired matrix size. To explicitly set the matrix size, add an HD5 attribute to the ``trajectory`` dataset named ``matrix``. This should be a 3 element 1D integer array with the nominal matrix size.

The trajectory is assumed to repeat for each slab and timepoint/volume.

Meta-Information
----------------

RIESLING is capable of storing additional meta-information and passing it through the processing chain. This should be stored in an HDF5 group named ``meta``, and consist of key-value pairs where the key is the dataset name and the value is a single floating-point number.
